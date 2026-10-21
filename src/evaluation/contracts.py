"""
Data Contracts for Evaluation Harness
GraphRAG v2.1

Defines schemas for experiments, results, and metrics.
Every evaluation run is fully traceable: what config was used,
what data version, what metrics, and what errors occurred.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    SINGLE_HOP = "single-hop"
    MULTI_HOP = "multi-hop"
    GLOBAL = "global"
    UNANSWERABLE = "unanswerable"
    NOISY = "noisy"


class RetrievalStrategy(str, Enum):
    NAIVE = "naive"      # Vector-only
    GRAPH = "graph"      # Graph-only
    HYBRID = "hybrid"    # Vector + Graph


class EvalErrorType(str, Enum):
    """Categorize WHY a wrong answer occurred — not just THAT it was wrong."""
    RETRIEVAL_MISS = "retrieval_miss"           # Relevant doc not found
    CONTEXT_TRUNCATED = "context_truncated"     # Chunk cut off key info
    GRAPH_EXTRACTION_FAIL = "graph_extraction"  # Graph didn't extract the right edges
    LLM_HALLUCINATION = "llm_hallucination"     # LLM made up info not in context
    LLM_OMISSION = "llm_omission"              # LLM missed info that WAS in context
    WRONG_REFUSAL = "wrong_refusal"             # Said "I don't know" when answer was available
    CORRECT_REFUSAL = "correct_refusal"         # Correctly said "I don't know" for unanswerable
    NONE = "none"                               # Answer was correct


class GroundTruthItem(BaseModel):
    """Schema for a single test question."""
    id: str
    type: QuestionType
    question: str
    ground_truth: str
    required_edges: list[str] = Field(default_factory=list)
    expected_behavior: Optional[str] = None


class ExperimentConfig(BaseModel):
    """Full configuration of an evaluation run — for reproducibility."""
    experiment_id: str
    dataset_version: str        # Hash of ground_truth.json
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "gemini-3.6-flash"
    top_k: int = 3
    chunk_size: int = 512
    prompt_version: str = "v1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str = ""


class QuestionResult(BaseModel):
    """Result of evaluating a single question with a single strategy."""
    experiment_id: str
    question_id: str
    question_type: QuestionType
    strategy: RetrievalStrategy

    # Retrieval metrics
    context_precision: Optional[float] = None   # 0.0 - 1.0
    graph_recall: Optional[float] = None        # Did we find required edges?
    retrieval_latency_ms: float = 0.0

    # Generation metrics
    faithfulness: Optional[float] = None        # 0.0 - 1.0
    answer_relevance: Optional[float] = None    # 0.0 - 1.0
    answer_correctness: Optional[float] = None  # 0.0 - 1.0
    generation_latency_ms: float = 0.0
    token_count: int = 0

    # Error analysis
    error_type: EvalErrorType = EvalErrorType.NONE

    # Raw data for debugging
    raw_answer: str = ""
    raw_context: list[str] = Field(default_factory=list)
