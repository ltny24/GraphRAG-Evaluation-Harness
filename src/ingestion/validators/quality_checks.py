"""
Data Quality Framework
GraphRAG v2.1 — Ingestion

Validates data BEFORE it enters Neo4j/Qdrant.
This is the "Quality Gate" — the pipeline STOPS if critical checks fail,
instead of silently loading garbage data into the databases.

Six quality dimensions checked:
1. Completeness  — No empty fields
2. Uniqueness    — No duplicate IDs
3. Validity      — Values within allowed ranges
4. Consistency   — Relations point to existing entities
5. Referential Integrity — Chunks belong to known documents
6. Freshness     — Detected via file_hash comparison (handled in pipeline.py)
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.ingestion.contracts import (
    ChunkRecord, EntityRecord, RelationRecord,
    GraphExtractionResult, EntityType
)


class Severity(str, Enum):
    """Error severity determines pipeline behavior."""
    WARNING = "warning"    # Log but continue
    CRITICAL = "critical"  # Block loading, send to DLQ


@dataclass
class QualityIssue:
    """A single data quality problem found during validation."""
    check_name: str
    severity: Severity
    message: str
    document_id: Optional[str] = None
    field: Optional[str] = None


@dataclass
class QualityReport:
    """Aggregated result of all quality checks on a batch of data."""
    document_id: str
    issues: list[QualityIssue] = field(default_factory=list)
    chunks_checked: int = 0
    entities_checked: int = 0
    relations_checked: int = 0

    @property
    def passed(self) -> bool:
        """Pipeline proceeds only if zero CRITICAL issues."""
        return not any(i.severity == Severity.CRITICAL for i in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.CRITICAL)

    def summary(self) -> str:
        status = "PASSED ✓" if self.passed else "FAILED ✗"
        return (f"Quality Report [{status}] doc={self.document_id}: "
                f"{self.chunks_checked} chunks, "
                f"{self.entities_checked} entities, "
                f"{self.relations_checked} relations | "
                f"{self.warning_count} warnings, {self.critical_count} critical")


class DataQualityValidator:
    """
    Runs all 5 applicable quality checks on extraction results.
    (Freshness is handled at pipeline level via hash comparison.)

    Usage:
        validator = DataQualityValidator()
        report = validator.validate(document_id, chunks, extraction)
        if report.passed:
            # proceed to loading
        else:
            # send to DLQ
    """

    def validate(
        self,
        document_id: str,
        chunks: list[ChunkRecord],
        extraction: GraphExtractionResult,
    ) -> QualityReport:
        """Run all quality checks and return a report."""
        report = QualityReport(document_id=document_id)
        report.chunks_checked = len(chunks)
        report.entities_checked = len(extraction.entities)
        report.relations_checked = len(extraction.relations)

        self._check_completeness(chunks, extraction, report)
        self._check_uniqueness(chunks, extraction, report)
        self._check_validity(extraction, report)
        self._check_consistency(extraction, report)
        self._check_referential_integrity(chunks, document_id, report)

        return report

    def _check_completeness(
        self, chunks: list[ChunkRecord],
        extraction: GraphExtractionResult,
        report: QualityReport
    ):
        """Check 1: No empty or whitespace-only fields."""
        for chunk in chunks:
            if not chunk.content.strip():
                report.issues.append(QualityIssue(
                    check_name="completeness",
                    severity=Severity.CRITICAL,
                    message=f"Chunk {chunk.chunk_id} has empty content",
                    document_id=report.document_id,
                    field="content"
                ))

        for entity in extraction.entities:
            if not entity.name.strip():
                report.issues.append(QualityIssue(
                    check_name="completeness",
                    severity=Severity.CRITICAL,
                    message="Entity has empty name",
                    document_id=report.document_id,
                    field="entity.name"
                ))

        for rel in extraction.relations:
            if not rel.source.strip() or not rel.target.strip():
                report.issues.append(QualityIssue(
                    check_name="completeness",
                    severity=Severity.CRITICAL,
                    message=f"Relation has empty source or target: {rel.source} -> {rel.target}",
                    document_id=report.document_id,
                    field="relation.source/target"
                ))

    def _check_uniqueness(
        self, chunks: list[ChunkRecord],
        extraction: GraphExtractionResult,
        report: QualityReport
    ):
        """Check 2: No duplicate chunk IDs or entity names within a document."""
        chunk_ids = [c.chunk_id for c in chunks]
        seen_ids = set()
        for cid in chunk_ids:
            if cid in seen_ids:
                report.issues.append(QualityIssue(
                    check_name="uniqueness",
                    severity=Severity.WARNING,
                    message=f"Duplicate chunk_id: {cid}",
                    document_id=report.document_id,
                    field="chunk_id"
                ))
            seen_ids.add(cid)

        # Entity dedup check (same name + type = duplicate)
        entity_keys = [(e.name.lower(), e.type) for e in extraction.entities]
        seen_entities = set()
        for key in entity_keys:
            if key in seen_entities:
                report.issues.append(QualityIssue(
                    check_name="uniqueness",
                    severity=Severity.WARNING,
                    message=f"Duplicate entity: {key[0]} ({key[1]})",
                    document_id=report.document_id,
                    field="entity"
                ))
            seen_entities.add(key)

    def _check_validity(
        self, extraction: GraphExtractionResult,
        report: QualityReport
    ):
        """Check 3: Entity types within allowed enum values."""
        allowed_types = {e.value for e in EntityType}
        for entity in extraction.entities:
            if entity.type.value not in allowed_types:
                report.issues.append(QualityIssue(
                    check_name="validity",
                    severity=Severity.WARNING,
                    message=f"Entity '{entity.name}' has unknown type: {entity.type}",
                    document_id=report.document_id,
                    field="entity.type"
                ))

        # Relation type should be uppercase and alphanumeric
        for rel in extraction.relations:
            sanitized = "".join(c for c in rel.type if c.isalnum() or c == "_")
            if sanitized != rel.type:
                report.issues.append(QualityIssue(
                    check_name="validity",
                    severity=Severity.WARNING,
                    message=f"Relation type '{rel.type}' contains invalid characters",
                    document_id=report.document_id,
                    field="relation.type"
                ))

    def _check_consistency(
        self, extraction: GraphExtractionResult,
        report: QualityReport
    ):
        """Check 4: Every relation's source and target must exist as an entity.
        This is the most critical check — broken references = broken graph.
        """
        entity_names = {e.name.lower() for e in extraction.entities}

        for rel in extraction.relations:
            if rel.source.lower() not in entity_names:
                report.issues.append(QualityIssue(
                    check_name="consistency",
                    severity=Severity.CRITICAL,
                    message=f"Relation source '{rel.source}' not found in entities",
                    document_id=report.document_id,
                    field="relation.source"
                ))
            if rel.target.lower() not in entity_names:
                report.issues.append(QualityIssue(
                    check_name="consistency",
                    severity=Severity.CRITICAL,
                    message=f"Relation target '{rel.target}' not found in entities",
                    document_id=report.document_id,
                    field="relation.target"
                ))

    def _check_referential_integrity(
        self, chunks: list[ChunkRecord],
        document_id: str,
        report: QualityReport
    ):
        """Check 5: All chunks reference the correct parent document."""
        for chunk in chunks:
            if chunk.document_id != document_id:
                report.issues.append(QualityIssue(
                    check_name="referential_integrity",
                    severity=Severity.CRITICAL,
                    message=f"Chunk {chunk.chunk_id} references document '{chunk.document_id}' "
                            f"but expected '{document_id}'",
                    document_id=report.document_id,
                    field="chunk.document_id"
                ))
