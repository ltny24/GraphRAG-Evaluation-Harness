"""
Data Contracts for Ingestion Pipeline
GraphRAG v2.1

Defines strict Pydantic schemas for data flowing between pipeline steps.
Each step's output MUST conform to these contracts before passing downstream.
This is the "Data Contract" pattern — if a step changes its output format,
the contract will catch it immediately instead of letting bad data propagate.
"""
import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ─── Enums ──────────────────────────────────────────────────────────────────

class ProcessingStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class EntityType(str, Enum):
    DRUG = "Drug"
    ENZYME = "Enzyme"
    DISEASE = "Disease"
    PROTEIN = "Protein"
    CHEMICAL = "Chemical"
    OTHER = "Other"


class ErrorType(str, Enum):
    """Categorize errors for meaningful analysis, not just generic 'failed'."""
    TIMEOUT = "timeout"                     # API timeout — retryable
    INVALID_JSON = "invalid_json"           # LLM returned garbage — retryable with different prompt
    SCHEMA_VIOLATION = "schema_violation"   # Data doesn't match contract — retryable
    REFERENTIAL_INTEGRITY = "ref_integrity" # Relation points to non-existent entity — NOT retryable
    CORRUPT_FILE = "corrupt_file"           # Source file unreadable — NOT retryable
    RATE_LIMIT = "rate_limit"               # API quota exceeded — retryable after delay
    UNKNOWN = "unknown"

    @classmethod
    def is_retryable(cls, error_type: "ErrorType") -> bool:
        """Distinguish transient errors from data errors."""
        return error_type in {cls.TIMEOUT, cls.INVALID_JSON, cls.SCHEMA_VIOLATION, cls.RATE_LIMIT}


# ─── Document Contract ──────────────────────────────────────────────────────

class DocumentRecord(BaseModel):
    """
    Tracks a source document through the pipeline.
    This is the foundation of Data Lineage — every piece of downstream data
    can be traced back to a specific document version.
    """
    document_id: str = Field(description="Unique ID, typically PMID for PubMed articles")
    source_path: str = Field(description="Path to the raw source file")
    file_hash: str = Field(description="SHA-256 hash of raw content for change detection")
    ingestion_version: str = Field(default="v2.1", description="Pipeline version that processed this")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    chunk_count: int = Field(default=0)
    entity_count: int = Field(default=0)
    relation_count: int = Field(default=0)

    @staticmethod
    def compute_hash(content: str) -> str:
        """Deterministic hash for idempotency checks."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ─── Chunk Contract ─────────────────────────────────────────────────────────

class ChunkRecord(BaseModel):
    """
    A text chunk with full traceability to its source document.
    The content_hash enables deduplication and change detection at chunk level.
    """
    chunk_id: str = Field(description="Format: {document_id}_chunk{index}")
    document_id: str = Field(description="FK to DocumentRecord")
    content: str = Field(min_length=1, description="Chunk text, must not be empty")
    chunk_index: int = Field(ge=0)
    content_hash: str = Field(description="SHA-256 of content for dedup")
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)

    @field_validator("content")
    @classmethod
    def content_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk content cannot be only whitespace")
        return v

    @staticmethod
    def compute_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ─── Entity & Relation Contracts ────────────────────────────────────────────

class EntityRecord(BaseModel):
    """An extracted biomedical entity with provenance tracking."""
    name: str = Field(min_length=1)
    type: EntityType
    source_chunk_id: str = Field(description="Which chunk this was extracted from")


class RelationRecord(BaseModel):
    """
    An extracted relationship between two entities.
    Both source and target MUST exist as EntityRecords — this is
    checked by the Data Quality validator, not here (cross-field validation).
    """
    source: str = Field(min_length=1, description="Source entity name")
    target: str = Field(min_length=1, description="Target entity name")
    type: str = Field(min_length=1, description="Relationship type e.g. TREATS, METABOLIZES")
    source_chunk_id: str = Field(description="Provenance: which chunk this came from")


class GraphExtractionResult(BaseModel):
    """Output of the LLM entity/relation extraction step."""
    entities: list[EntityRecord] = Field(default_factory=list)
    relations: list[RelationRecord] = Field(default_factory=list)


# ─── Pipeline Run Contract ──────────────────────────────────────────────────

class PipelineRunRecord(BaseModel):
    """
    Observability record for each pipeline execution.
    Answers: "What happened in this run? How many files? How long? What failed?"
    """
    run_id: str = Field(description="Format: run_YYYYMMDD_NNN")
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    status: ProcessingStatus = Field(default=ProcessingStatus.PROCESSING)
    total_files: int = 0
    files_succeeded: int = 0
    files_failed: int = 0
    chunks_created: int = 0
    entities_extracted: int = 0
    relations_extracted: int = 0
    duration_seconds: Optional[float] = None
    error_summary: Optional[str] = None


# ─── Dead Letter Queue Record ───────────────────────────────────────────────

class DLQRecord(BaseModel):
    """
    Failed data goes here instead of silently disappearing.
    The error_type field is critical — it tells us whether retrying makes sense.
    """
    run_id: str
    document_id: str
    stage: str = Field(description="Pipeline stage: chunking | extraction | validation | loading")
    error_type: ErrorType
    error_message: str
    payload: Optional[str] = Field(default=None, description="JSON dump of the failed data")
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
