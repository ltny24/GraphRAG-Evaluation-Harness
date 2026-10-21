"""
Evaluation Metrics
GraphRAG v2.1

Three core metrics + error classification, all using LLM-as-Judge.

Metrics:
1. Faithfulness     — Is the answer grounded in the context? (No hallucination)
2. Context Precision — Does the retrieved context contain the answer?
3. Answer Relevance  — Does the answer actually address the question?
4. Answer Correctness — Does the answer match the ground truth? (Requires reference)

Error Classification:
- Categorizes WHY a wrong answer occurred, not just THAT it was wrong.
- Enables root cause analysis: "GraphRAG fails on multi-hop because of retrieval_miss"

Design note on LLM-as-Judge reliability:
- We acknowledge it's imperfect (as stated in the project requirements)
- We use structured output (JSON) to reduce judge variance
- The harness supports spot-checking judge decisions manually
"""
import json
import logging
from pydantic import BaseModel, Field
from src.infrastructure.llm_client import LLMClient
from src.evaluation.contracts import EvalErrorType

logger = logging.getLogger(__name__)


# ─── Structured Output Schemas for Judge ────────────────────────────────

class FaithfulnessScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="0.0 = full hallucination, 1.0 = fully grounded")
    reasoning: str = Field(description="Brief explanation of the score")

class ContextPrecisionScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="0.0 = context irrelevant, 1.0 = context contains answer")
    reasoning: str = Field(description="Brief explanation")

class AnswerRelevanceScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="0.0 = off-topic, 1.0 = directly addresses question")
    reasoning: str = Field(description="Brief explanation")

class CorrectnessScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="0.0 = completely wrong, 1.0 = matches ground truth")
    reasoning: str = Field(description="Brief explanation")


class MetricsEvaluator:
    """Evaluates RAG answers using LLM-as-Judge with structured output."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def score_faithfulness(self, answer: str, context: list[str]) -> tuple[float, str]:
        """Is the answer grounded in the context? (Hallucination check)"""
        ctx_str = "\n".join(context) if context else "(empty)"
        prompt = f"""You are an expert evaluator. Score the faithfulness of the Answer.
Faithfulness = the answer contains ONLY information that can be found in the Context.
If the answer contains claims not supported by the context, score lower.

Context:
{ctx_str}

Answer:
{answer}

Score from 0.0 (full hallucination) to 1.0 (fully grounded in context). Provide brief reasoning."""

        try:
            parsed, _ = self.llm.generate_structured(prompt, FaithfulnessScore)
            return parsed.score, parsed.reasoning
        except Exception as e:
            logger.warning(f"Faithfulness scoring failed: {e}")
            return 0.0, f"Scoring error: {e}"

    def score_context_precision(self, question: str, context: list[str],
                                 ground_truth: str) -> tuple[float, str]:
        """Does the retrieved context contain information needed to answer correctly?"""
        ctx_str = "\n".join(context) if context else "(empty)"
        prompt = f"""You are an expert evaluator. Score the context precision.
Context Precision = does the retrieved Context contain the information needed to produce the Ground Truth answer?

Question: {question}
Ground Truth: {ground_truth}
Retrieved Context:
{ctx_str}

Score from 0.0 (context has nothing useful) to 1.0 (context fully supports the ground truth). Provide brief reasoning."""

        try:
            parsed, _ = self.llm.generate_structured(prompt, ContextPrecisionScore)
            return parsed.score, parsed.reasoning
        except Exception as e:
            logger.warning(f"Context precision scoring failed: {e}")
            return 0.0, f"Scoring error: {e}"

    def score_answer_relevance(self, question: str, answer: str) -> tuple[float, str]:
        """Does the answer address the question asked?"""
        prompt = f"""You are an expert evaluator. Score the answer relevance.
Answer Relevance = does the Answer directly address the Question?
An answer that is technically correct but doesn't answer what was asked scores lower.

Question: {question}
Answer: {answer}

Score from 0.0 (completely off-topic) to 1.0 (directly addresses the question). Provide brief reasoning."""

        try:
            parsed, _ = self.llm.generate_structured(prompt, AnswerRelevanceScore)
            return parsed.score, parsed.reasoning
        except Exception as e:
            logger.warning(f"Answer relevance scoring failed: {e}")
            return 0.0, f"Scoring error: {e}"

    def score_correctness(self, question: str, answer: str,
                           ground_truth: str) -> tuple[float, str]:
        """Does the answer match the ground truth?"""
        prompt = f"""You are an expert evaluator. Score the correctness of the Answer.
Compare the Answer against the Ground Truth. The answer doesn't need to be word-for-word identical,
but it must convey the same key information.

Question: {question}
Ground Truth: {ground_truth}
Answer: {answer}

Score from 0.0 (completely wrong) to 1.0 (matches ground truth perfectly). Provide brief reasoning."""

        try:
            parsed, _ = self.llm.generate_structured(prompt, CorrectnessScore)
            return parsed.score, parsed.reasoning
        except Exception as e:
            logger.warning(f"Correctness scoring failed: {e}")
            return 0.0, f"Scoring error: {e}"

    def classify_error(self, question: str, answer: str, context: list[str],
                        ground_truth: str, question_type: str) -> EvalErrorType:
        """Classify WHY an answer was wrong — for root cause analysis."""
        # Quick heuristics before burning an LLM call
        answer_lower = answer.lower()

        # Check for refusal
        refusal_phrases = ["i don't have", "i don't know", "not enough information",
                           "cannot answer", "no information"]
        is_refusal = any(p in answer_lower for p in refusal_phrases)

        if question_type == "unanswerable":
            return EvalErrorType.CORRECT_REFUSAL if is_refusal else EvalErrorType.LLM_HALLUCINATION

        if is_refusal:
            return EvalErrorType.WRONG_REFUSAL

        # Check if context was empty or irrelevant
        if not context or all(len(c.strip()) < 10 for c in context):
            return EvalErrorType.RETRIEVAL_MISS

        # Check if ground truth info appears in context but not in answer
        gt_keywords = set(ground_truth.lower().split()) - {"the", "a", "an", "is", "are", "and", "of", "to", "in"}
        ctx_text = " ".join(context).lower()
        gt_in_context = sum(1 for kw in gt_keywords if kw in ctx_text) / max(len(gt_keywords), 1)

        if gt_in_context < 0.3:
            return EvalErrorType.RETRIEVAL_MISS
        elif gt_in_context >= 0.3:
            # Context has the info but answer is wrong
            answer_has_gt = sum(1 for kw in gt_keywords if kw in answer_lower) / max(len(gt_keywords), 1)
            if answer_has_gt < 0.3:
                return EvalErrorType.LLM_OMISSION

        return EvalErrorType.NONE
