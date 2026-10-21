"""
Data Ingestion Pipeline
GraphRAG v2.1

Orchestrates the full ETL flow with:
- Idempotency: Skips files that haven't changed (via content hash)
- Checkpointing: Resumes from last successful step after crashes
- Data Quality Gate: Blocks bad data from reaching databases
- Dead Letter Queue: Preserves failed data for analysis
- Lineage: Every piece of data traceable back to source file + version
- Observability: Full telemetry on each run

Usage:
    python -m src.ingestion.pipeline
"""
import json
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.contracts import (
    DocumentRecord, ChunkRecord, EntityRecord, RelationRecord,
    GraphExtractionResult, PipelineRunRecord, DLQRecord,
    ProcessingStatus, ErrorType, EntityType
)
from src.ingestion.validators.quality_checks import DataQualityValidator
from src.infrastructure.database import MetadataStore
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.telemetry import PipelineTelemetry

# Reuse existing chunking logic
from src.ingestion.chunker import chunk_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Pydantic schema for LLM extraction (same as extractor.py but with provenance) ─
from pydantic import BaseModel, Field

class _LLMEntity(BaseModel):
    name: str = Field(description="Entity name")
    type: str = Field(description="Category: Drug, Enzyme, Disease, Protein, Chemical, Other")

class _LLMRelation(BaseModel):
    source: str
    target: str
    type: str

class _LLMExtractionResult(BaseModel):
    entities: list[_LLMEntity]
    relations: list[_LLMRelation]


class IngestionPipeline:
    """
    Production-grade data ingestion pipeline.
    
    Key properties:
    - Idempotent: Running twice with same data = no effect
    - Resumable: Crashes at step N → restart continues from step N
    - Observable: Every run is recorded with full metrics
    """

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.db = MetadataStore(self.data_dir / "metadata.db")
        self.llm = LLMClient(model_id="gemini-3.6-flash", max_retries=3)
        self.validator = DataQualityValidator()

        # Pipeline config (could be loaded from .env)
        self.chunk_size = 512
        self.chunk_overlap = 64
        self.extraction_limit = 3  # Keep small for free tier, but pipeline handles any N

    def _generate_run_id(self) -> str:
        now = datetime.now(timezone.utc)
        # Check existing runs today
        existing = self.db.get_run_history(limit=100)
        today_runs = [r for r in existing if r["run_id"].startswith(f"run_{now.strftime('%Y%m%d')}")]
        seq = len(today_runs) + 1
        return f"run_{now.strftime('%Y%m%d')}_{seq:03d}"

    def run(self):
        """Execute the full pipeline."""
        run_id = self._generate_run_id()
        telemetry = PipelineTelemetry(run_id)

        logger.info(f"{'='*60}")
        logger.info(f"PIPELINE RUN: {run_id}")
        logger.info(f"{'='*60}")

        # ─── Step 1: Discover documents ────────────────────────────────
        step = telemetry.start_step("discover_documents")
        documents = self._discover_documents()
        step.items_processed = len(documents)
        step.stop()
        logger.info(f"[1/5] Discovered {len(documents)} documents in data/raw/")

        # ─── Step 2: Filter changed documents (Idempotency) ───────────
        step = telemetry.start_step("filter_changed")
        to_process = self._filter_changed(documents)
        step.items_processed = len(to_process)
        step.stop()
        logger.info(f"[2/5] {len(to_process)} documents need processing "
                     f"({len(documents) - len(to_process)} skipped — unchanged)")

        if not to_process:
            logger.info("Nothing to do. Pipeline is up to date.")
            telemetry.print_report()
            return

        # Create pipeline run record
        run_record = PipelineRunRecord(
            run_id=run_id,
            total_files=len(to_process)
        )
        self.db.create_run(run_record)

        # ─── Step 3-5: Process each document ──────────────────────────
        total_chunks = 0
        total_entities = 0
        total_relations = 0
        files_ok = 0
        files_fail = 0

        for i, (doc_id, doc_data) in enumerate(to_process.items()):
            logger.info(f"\n[3/5] Processing document {i+1}/{len(to_process)}: {doc_id}")

            try:
                # Mark as PROCESSING
                self.db.upsert_document(DocumentRecord(
                    document_id=doc_id,
                    source_path=str(doc_data["source_path"]),
                    file_hash=doc_data["file_hash"],
                    processing_status=ProcessingStatus.PROCESSING,
                ))

                # Step 3: Chunk
                step_chunk = telemetry.start_step(f"chunk_{doc_id}")
                chunks = self._chunk_document(doc_id, doc_data)
                step_chunk.items_processed = len(chunks)
                step_chunk.stop()
                logger.info(f"  → Chunked into {len(chunks)} chunks")

                # Step 4: Extract entities & relations (LLM)
                step_extract = telemetry.start_step(f"extract_{doc_id}")
                extraction = self._extract_graph(doc_id, chunks, step_extract)
                step_extract.stop()
                logger.info(f"  → Extracted {len(extraction.entities)} entities, "
                             f"{len(extraction.relations)} relations")

                # Step 5: Validate (Quality Gate)
                step_validate = telemetry.start_step(f"validate_{doc_id}")
                quality_report = self.validator.validate(doc_id, chunks, extraction)
                step_validate.stop()
                logger.info(f"  → Quality: {quality_report.summary()}")

                if not quality_report.passed:
                    # CRITICAL issues found → DLQ
                    logger.warning(f"  ✗ Quality gate FAILED for {doc_id}. Sending to DLQ.")
                    self.db.add_to_dlq(DLQRecord(
                        run_id=run_id,
                        document_id=doc_id,
                        stage="validation",
                        error_type=ErrorType.SCHEMA_VIOLATION,
                        error_message=quality_report.summary(),
                        payload=json.dumps([i.message for i in quality_report.issues]),
                    ))
                    self.db.update_document_status(doc_id, ProcessingStatus.FAILED)
                    files_fail += 1
                    continue

                # Quality passed → update lineage
                self.db.update_document_status(
                    doc_id, ProcessingStatus.COMPLETED,
                    chunk_count=len(chunks),
                    entity_count=len(extraction.entities),
                    relation_count=len(extraction.relations),
                )
                total_chunks += len(chunks)
                total_entities += len(extraction.entities)
                total_relations += len(extraction.relations)
                files_ok += 1

            except Exception as e:
                logger.error(f"  ✗ Failed to process {doc_id}: {e}")
                error_type = self._classify_error(e)
                self.db.add_to_dlq(DLQRecord(
                    run_id=run_id,
                    document_id=doc_id,
                    stage="processing",
                    error_type=error_type,
                    error_message=str(e),
                ))
                self.db.update_document_status(doc_id, ProcessingStatus.FAILED)
                files_fail += 1

        # ─── Finalize Run ─────────────────────────────────────────────
        final_status = ProcessingStatus.COMPLETED if files_fail == 0 else ProcessingStatus.FAILED
        self.db.complete_run(
            run_id=run_id,
            status=final_status,
            files_succeeded=files_ok,
            files_failed=files_fail,
            chunks_created=total_chunks,
            entities_extracted=total_entities,
            relations_extracted=total_relations,
            duration_seconds=telemetry.total_duration_seconds,
            error_summary=f"{files_fail} files failed" if files_fail > 0 else None,
        )

        telemetry.print_report()
        self._print_run_summary(run_id, files_ok, files_fail, total_chunks,
                                 total_entities, total_relations)

    # ─── Internal Methods ────────────────────────────────────────────────

    def _discover_documents(self) -> dict:
        """Scan raw data directory and compute file hashes."""
        documents = {}
        raw_file = self.raw_dir / "pubmed_ddi.json"
        if raw_file.exists():
            content = raw_file.read_text(encoding="utf-8")
            file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            articles = json.loads(content)
            for article in articles:
                pmid = article["pmid"]
                article_content = json.dumps(article, ensure_ascii=False)
                documents[pmid] = {
                    "source_path": str(raw_file),
                    "file_hash": hashlib.sha256(article_content.encode("utf-8")).hexdigest(),
                    "title": article.get("title", ""),
                    "abstract": article.get("abstract", ""),
                }
        return documents

    def _filter_changed(self, documents: dict) -> dict:
        """Idempotency: only process documents that are new or changed."""
        to_process = {}
        for doc_id, doc_data in documents.items():
            if self.db.has_document_changed(doc_id, doc_data["file_hash"]):
                to_process[doc_id] = doc_data
        return to_process

    def _chunk_document(self, doc_id: str, doc_data: dict) -> list[ChunkRecord]:
        """Chunk a single document into ChunkRecords with hashes."""
        full_text = f"Title: {doc_data['title']}\n\nAbstract: {doc_data['abstract']}"
        raw_chunks = chunk_text(full_text, self.chunk_size, self.chunk_overlap)

        records = []
        for idx, (text, start, end) in enumerate(raw_chunks):
            records.append(ChunkRecord(
                chunk_id=f"{doc_id}_chunk{idx}",
                document_id=doc_id,
                content=text,
                chunk_index=idx,
                content_hash=ChunkRecord.compute_hash(text),
                char_start=start,
                char_end=end,
            ))
        return records

    def _extract_graph(
        self, doc_id: str, chunks: list[ChunkRecord], step_metrics
    ) -> GraphExtractionResult:
        """Extract entities and relations from chunks using LLM."""
        all_entities = []
        all_relations = []

        # Respect extraction limit (API quota)
        chunks_to_process = chunks[:self.extraction_limit]

        for chunk in chunks_to_process:
            prompt = f"""You are a medical data extraction expert.
Analyze the following text and extract all relevant biomedical entities and relationships.

Text:
\"\"\"{chunk.content}\"\"\"

Focus on: Drugs, Enzymes (CYP3A4, P-gp), Diseases, and how they interact."""

            try:
                parsed, metadata = self.llm.generate_structured(
                    prompt, schema=_LLMExtractionResult, temperature=0.1
                )
                step_metrics.input_tokens += metadata.get("input_tokens", 0)
                step_metrics.output_tokens += metadata.get("output_tokens", 0)

                # Convert to contract types with provenance
                for ent in parsed.entities:
                    try:
                        entity_type = EntityType(ent.type)
                    except ValueError:
                        entity_type = EntityType.OTHER

                    all_entities.append(EntityRecord(
                        name=ent.name,
                        type=entity_type,
                        source_chunk_id=chunk.chunk_id,
                    ))

                for rel in parsed.relations:
                    all_relations.append(RelationRecord(
                        source=rel.source,
                        target=rel.target,
                        type=rel.type,
                        source_chunk_id=chunk.chunk_id,
                    ))

                step_metrics.items_processed += 1

            except Exception as e:
                logger.warning(f"  Extraction failed for chunk {chunk.chunk_id}: {e}")
                step_metrics.errors += 1

        return GraphExtractionResult(entities=all_entities, relations=all_relations)

    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify errors for meaningful DLQ analysis."""
        error_str = str(error).lower()
        if "429" in error_str or "rate" in error_str:
            return ErrorType.RATE_LIMIT
        elif "timeout" in error_str:
            return ErrorType.TIMEOUT
        elif "json" in error_str:
            return ErrorType.INVALID_JSON
        elif "schema" in error_str or "validation" in error_str:
            return ErrorType.SCHEMA_VIOLATION
        return ErrorType.UNKNOWN

    def _print_run_summary(self, run_id, ok, fail, chunks, entities, relations):
        logger.info(f"\n{'='*60}")
        logger.info(f"RUN COMPLETE: {run_id}")
        logger.info(f"  Files: {ok} succeeded, {fail} failed")
        logger.info(f"  Data:  {chunks} chunks, {entities} entities, {relations} relations")
        dlq = self.db.get_unresolved_dlq(run_id)
        if dlq:
            logger.info(f"  DLQ:   {len(dlq)} unresolved errors")
            for item in dlq:
                logger.info(f"    → [{item['error_type']}] {item['document_id']}: {item['error_message'][:80]}")
        # Show lineage
        history = self.db.get_run_history(limit=5)
        if history:
            logger.info(f"\n  Recent Pipeline Runs:")
            for h in history:
                logger.info(f"    {h['run_id']} | {h['status']} | "
                             f"{h['files_succeeded']}/{h['total_files']} files | "
                             f"{h.get('duration_seconds', 0):.1f}s")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()
