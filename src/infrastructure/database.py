"""
SQLite Metadata Store
GraphRAG v2.1 — Infrastructure

Central database for:
- Data Lineage: Track every document from raw → processed → loaded
- Pipeline Observability: Record each run's statistics and errors
- Dead Letter Queue: Preserve failed data for analysis and retry
- Experiment Tracking: Store evaluation results for comparison

Why SQLite?
- Zero setup (built into Python stdlib)
- ACID transactions (data integrity during pipeline crashes)
- Single file (easy to version, backup, inspect)
- SQL queries for analysis (vs parsing JSON/CSV manually)
"""
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.ingestion.contracts import (
    DocumentRecord, PipelineRunRecord, DLQRecord,
    ProcessingStatus, ErrorType
)


class MetadataStore:
    """SQLite-backed metadata store for pipeline lineage and observability."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "metadata.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self):
        """Create all tables if they don't exist. Idempotent by design."""
        self.conn.executescript("""
            -- Documents: Data Lineage tracking
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                ingestion_version TEXT DEFAULT 'v2.1',
                created_at TEXT NOT NULL,
                updated_at TEXT,
                processing_status TEXT DEFAULT 'PENDING',
                chunk_count INTEGER DEFAULT 0,
                entity_count INTEGER DEFAULT 0,
                relation_count INTEGER DEFAULT 0
            );

            -- Pipeline Runs: Observability
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT DEFAULT 'PROCESSING',
                total_files INTEGER DEFAULT 0,
                files_succeeded INTEGER DEFAULT 0,
                files_failed INTEGER DEFAULT 0,
                chunks_created INTEGER DEFAULT 0,
                entities_extracted INTEGER DEFAULT 0,
                relations_extracted INTEGER DEFAULT 0,
                duration_seconds REAL,
                error_summary TEXT
            );

            -- Dead Letter Queue: Failed data preservation
            CREATE TABLE IF NOT EXISTS dead_letter_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                error_type TEXT NOT NULL,
                error_message TEXT NOT NULL,
                payload TEXT,
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
            );

            -- Experiments: Evaluation tracking
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                dataset_version TEXT,
                retrieval_strategy TEXT,
                embedding_model TEXT,
                llm_model TEXT,
                top_k INTEGER,
                chunk_size INTEGER,
                prompt_version TEXT,
                created_at TEXT NOT NULL,
                notes TEXT
            );

            -- Question Results: Per-question evaluation metrics
            CREATE TABLE IF NOT EXISTS question_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                question_type TEXT,
                strategy TEXT,
                context_precision REAL,
                faithfulness REAL,
                answer_relevance REAL,
                answer_correctness REAL,
                retrieval_latency_ms REAL,
                generation_latency_ms REAL,
                token_count INTEGER,
                error_type TEXT,
                raw_answer TEXT,
                raw_context TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            );
        """)
        self.conn.commit()

    # ─── Document Operations (Lineage) ──────────────────────────────────

    def get_document(self, document_id: str) -> Optional[dict]:
        """Check if a document has been processed before."""
        row = self.conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return dict(row) if row else None

    def upsert_document(self, doc: DocumentRecord):
        """Insert or update a document record. Core of idempotency."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute("""
            INSERT INTO documents (document_id, source_path, file_hash, ingestion_version,
                                   created_at, updated_at, processing_status, chunk_count,
                                   entity_count, relation_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                file_hash = excluded.file_hash,
                updated_at = ?,
                processing_status = excluded.processing_status,
                chunk_count = excluded.chunk_count,
                entity_count = excluded.entity_count,
                relation_count = excluded.relation_count
        """, (
            doc.document_id, doc.source_path, doc.file_hash, doc.ingestion_version,
            doc.created_at.isoformat(), now, doc.processing_status.value,
            doc.chunk_count, doc.entity_count, doc.relation_count,
            now
        ))
        self.conn.commit()

    def has_document_changed(self, document_id: str, new_hash: str) -> bool:
        """Check if a document's content has changed since last processing.
        Returns True if: document is new, or hash differs, or status is FAILED.
        """
        existing = self.get_document(document_id)
        if not existing:
            return True  # New document
        if existing["file_hash"] != new_hash:
            return True  # Content changed
        if existing["processing_status"] == ProcessingStatus.FAILED.value:
            return True  # Previously failed, should retry
        return False

    def update_document_status(self, document_id: str, status: ProcessingStatus,
                                chunk_count: int = 0, entity_count: int = 0,
                                relation_count: int = 0):
        """Update processing status and counts after pipeline step completes."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute("""
            UPDATE documents SET processing_status = ?, updated_at = ?,
                                 chunk_count = ?, entity_count = ?, relation_count = ?
            WHERE document_id = ?
        """, (status.value, now, chunk_count, entity_count, relation_count, document_id))
        self.conn.commit()

    # ─── Pipeline Run Operations (Observability) ────────────────────────

    def create_run(self, run: PipelineRunRecord):
        """Record the start of a pipeline run."""
        self.conn.execute("""
            INSERT INTO pipeline_runs (run_id, started_at, status, total_files)
            VALUES (?, ?, ?, ?)
        """, (run.run_id, run.started_at.isoformat(), run.status.value, run.total_files))
        self.conn.commit()

    def complete_run(self, run_id: str, status: ProcessingStatus,
                      files_succeeded: int, files_failed: int,
                      chunks_created: int, entities_extracted: int,
                      relations_extracted: int, duration_seconds: float,
                      error_summary: Optional[str] = None):
        """Finalize a pipeline run with statistics."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute("""
            UPDATE pipeline_runs SET
                completed_at = ?, status = ?,
                files_succeeded = ?, files_failed = ?,
                chunks_created = ?, entities_extracted = ?, relations_extracted = ?,
                duration_seconds = ?, error_summary = ?
            WHERE run_id = ?
        """, (now, status.value, files_succeeded, files_failed,
              chunks_created, entities_extracted, relations_extracted,
              duration_seconds, error_summary, run_id))
        self.conn.commit()

    def get_run_history(self, limit: int = 10) -> list[dict]:
        """Get recent pipeline runs for observability dashboard."""
        rows = self.conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ─── Dead Letter Queue Operations ───────────────────────────────────

    def add_to_dlq(self, record: DLQRecord):
        """Send failed data to the Dead Letter Queue."""
        self.conn.execute("""
            INSERT INTO dead_letter_queue
                (run_id, document_id, stage, error_type, error_message,
                 payload, retry_count, max_retries, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.run_id, record.document_id, record.stage,
            record.error_type.value, record.error_message,
            record.payload, record.retry_count, record.max_retries,
            record.created_at.isoformat()
        ))
        self.conn.commit()

    def get_unresolved_dlq(self, run_id: Optional[str] = None) -> list[dict]:
        """Get all unresolved DLQ entries, optionally filtered by run."""
        if run_id:
            rows = self.conn.execute(
                "SELECT * FROM dead_letter_queue WHERE resolved_at IS NULL AND run_id = ?",
                (run_id,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM dead_letter_queue WHERE resolved_at IS NULL"
            ).fetchall()
        return [dict(r) for r in rows]

    # ─── Experiment Operations ──────────────────────────────────────────

    def create_experiment(self, experiment_id: str, config: dict):
        """Record an evaluation experiment's configuration."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute("""
            INSERT INTO experiments
                (experiment_id, dataset_version, retrieval_strategy,
                 embedding_model, llm_model, top_k, chunk_size,
                 prompt_version, created_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id,
            config.get("dataset_version", ""),
            config.get("retrieval_strategy", ""),
            config.get("embedding_model", "all-MiniLM-L6-v2"),
            config.get("llm_model", "gemini-3.6-flash"),
            config.get("top_k", 3),
            config.get("chunk_size", 512),
            config.get("prompt_version", "v1"),
            now,
            config.get("notes", "")
        ))
        self.conn.commit()

    def save_question_result(self, result: dict):
        """Save a single question's evaluation result."""
        now = datetime.now(timezone.utc).isoformat()
        ctx_json = json.dumps(result.get("raw_context", []), ensure_ascii=False)
        self.conn.execute("""
            INSERT INTO question_results
                (experiment_id, question_id, question_type, strategy,
                 context_precision, faithfulness, answer_relevance, answer_correctness,
                 retrieval_latency_ms, generation_latency_ms, token_count,
                 error_type, raw_answer, raw_context, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result["experiment_id"], result["question_id"], result.get("question_type"),
            result["strategy"],
            result.get("context_precision"), result.get("faithfulness"),
            result.get("answer_relevance"), result.get("answer_correctness"),
            result.get("retrieval_latency_ms"), result.get("generation_latency_ms"),
            result.get("token_count"),
            result.get("error_type"), result.get("raw_answer"),
            ctx_json, now
        ))
        self.conn.commit()

    def get_experiment_results(self, experiment_id: str) -> list[dict]:
        """Retrieve all results for an experiment."""
        rows = self.conn.execute(
            "SELECT * FROM question_results WHERE experiment_id = ? ORDER BY question_id",
            (experiment_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ─── Cleanup ────────────────────────────────────────────────────────

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
