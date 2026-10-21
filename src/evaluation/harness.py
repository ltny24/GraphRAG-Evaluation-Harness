"""
Evaluation Harness
GraphRAG v2.1

The main evaluation runner that:
1. Loads the ground truth test set
2. Runs each question through all 3 strategies (Naive/Graph/Hybrid)
3. Scores with 4 metrics + error classification
4. Records everything in SQLite for experiment tracking
5. Prints a summary report

Usage:
    python -m src.evaluation.harness

Design decisions:
- Strategies are swappable (share same interface)
- Metrics are separated from the runner (single responsibility)
- Every result is persisted (no ephemeral evaluation)
- Rate limiting built-in (3s delay between LLM calls)
"""
import json
import hashlib
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.contracts import (
    GroundTruthItem, ExperimentConfig, QuestionResult,
    QuestionType, RetrievalStrategy, EvalErrorType
)
from src.evaluation.strategies.retrieval import get_strategies, BaseStrategy
from src.evaluation.metrics.scoring import MetricsEvaluator
from src.infrastructure.database import MetadataStore
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.telemetry import PipelineTelemetry

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Rate limit delay (seconds) between LLM calls
RATE_LIMIT_DELAY = 4


class EvaluationHarness:
    """
    Runs structured A/B evaluation across retrieval strategies.

    Key properties:
    - Reproducible: Every run records full config + data version
    - Comparable: Same questions, same metrics, different strategies
    - Analyzable: Error classification enables root cause analysis
    """

    def __init__(self):
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "eval"
        self.db = MetadataStore()
        self.llm = LLMClient(model_id="gemini-3.6-flash", max_retries=3)
        self.metrics = MetricsEvaluator(self.llm)

    def load_ground_truth(self) -> list[GroundTruthItem]:
        """Load the hand-curated test set."""
        gt_path = self.data_dir / "ground_truth.json"
        with open(gt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [GroundTruthItem(**item) for item in data]

    def _compute_dataset_version(self) -> str:
        """Hash the ground truth file for version tracking."""
        gt_path = self.data_dir / "ground_truth.json"
        content = gt_path.read_text(encoding="utf-8")
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _generate_experiment_id(self) -> str:
        now = datetime.now(timezone.utc)
        return f"exp_{now.strftime('%Y%m%d_%H%M%S')}"

    def run(self, strategies: list[str] = None, max_questions: int = None):
        """
        Execute the full evaluation.

        Args:
            strategies: List of strategy names to test. Default: all three.
            max_questions: Limit number of questions (for development/quota saving).
        """
        if strategies is None:
            strategies = ["naive", "graph", "hybrid"]

        experiment_id = self._generate_experiment_id()
        dataset_version = self._compute_dataset_version()
        telemetry = PipelineTelemetry(experiment_id)

        # Load test data
        ground_truth = self.load_ground_truth()
        if max_questions:
            ground_truth = ground_truth[:max_questions]

        logger.info("=" * 70)
        logger.info(f"EVALUATION HARNESS — Experiment: {experiment_id}")
        logger.info(f"  Dataset version: {dataset_version}")
        logger.info(f"  Questions: {len(ground_truth)}")
        logger.info(f"  Strategies: {strategies}")
        logger.info("=" * 70)

        # Record experiment config
        config = {
            "dataset_version": dataset_version,
            "retrieval_strategy": ",".join(strategies),
            "notes": f"Testing {len(ground_truth)} questions across {len(strategies)} strategies",
        }
        self.db.create_experiment(experiment_id, config)

        # Initialize strategies
        step_init = telemetry.start_step("init_strategies")
        active_strategies = get_strategies(strategies, self.llm)
        step_init.stop()
        for s_name in active_strategies:
            logger.info(f"  ✓ Initialized strategy: {s_name}")

        # Run evaluation
        all_results: list[QuestionResult] = []
        total = len(ground_truth) * len(strategies)
        current = 0

        for item in ground_truth:
            logger.info(f"\n{'─'*50}")
            logger.info(f"Q [{item.id}] ({item.type.value}): {item.question[:80]}...")

            for s_name, strategy in active_strategies.items():
                current += 1
                logger.info(f"  [{current}/{total}] Strategy: {s_name}")

                result = self._evaluate_single(
                    experiment_id, item, s_name, strategy, telemetry
                )
                all_results.append(result)

                # Save to SQLite immediately (crash-safe)
                self.db.save_question_result(result.model_dump())

                # Rate limiting
                time.sleep(RATE_LIMIT_DELAY)

        # Print summary
        telemetry.print_report()
        self._print_results_summary(experiment_id, all_results)

        # Cleanup
        for s_name, strategy in active_strategies.items():
            if hasattr(strategy, 'close'):
                strategy.close()

        return all_results

    def _evaluate_single(
        self,
        experiment_id: str,
        item: GroundTruthItem,
        strategy_name: str,
        strategy: BaseStrategy,
        telemetry: PipelineTelemetry,
    ) -> QuestionResult:
        """Evaluate a single question with a single strategy."""

        # ─── 1. Retrieve ───────────────────────────────────────────
        step_r = telemetry.start_step(f"retrieve_{strategy_name}_{item.id}")
        try:
            context = strategy.retrieve(item.question)
        except Exception as e:
            logger.warning(f"    Retrieval failed: {e}")
            context = []
        step_r.stop()
        retrieval_ms = step_r.duration_ms

        # ─── 2. Generate ──────────────────────────────────────────
        step_g = telemetry.start_step(f"generate_{strategy_name}_{item.id}")
        try:
            answer, gen_meta = strategy.generate(item.question, context)
        except Exception as e:
            logger.warning(f"    Generation failed: {e}")
            answer = f"Error: {e}"
            gen_meta = {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0}
        step_g.stop()
        generation_ms = step_g.duration_ms
        token_count = gen_meta.get("input_tokens", 0) + gen_meta.get("output_tokens", 0)

        logger.info(f"    Answer: {answer[:100]}...")

        # ─── 3. Score (if API available) ───────────────────────────
        faithfulness = None
        ctx_precision = None
        answer_relevance = None
        answer_correctness = None

        try:
            time.sleep(RATE_LIMIT_DELAY)
            f_score, _ = self.metrics.score_faithfulness(answer, context)
            faithfulness = f_score
            logger.info(f"    Faithfulness: {f_score:.2f}")

            time.sleep(RATE_LIMIT_DELAY)
            cp_score, _ = self.metrics.score_context_precision(
                item.question, context, item.ground_truth
            )
            ctx_precision = cp_score

            time.sleep(RATE_LIMIT_DELAY)
            ar_score, _ = self.metrics.score_answer_relevance(item.question, answer)
            answer_relevance = ar_score

            time.sleep(RATE_LIMIT_DELAY)
            ac_score, _ = self.metrics.score_correctness(
                item.question, answer, item.ground_truth
            )
            answer_correctness = ac_score
            logger.info(f"    Correctness: {ac_score:.2f}")

        except Exception as e:
            logger.warning(f"    Scoring failed (API quota?): {e}")

        # ─── 4. Error Classification ──────────────────────────────
        error_type = self.metrics.classify_error(
            item.question, answer, context, item.ground_truth, item.type.value
        )
        if error_type != EvalErrorType.NONE:
            logger.info(f"    Error type: {error_type.value}")

        # ─── 5. Graph Recall (for graph-aware strategies) ─────────
        graph_recall = None
        if item.required_edges and strategy_name in ("graph", "hybrid"):
            ctx_text = " ".join(context).lower()
            edges_found = sum(
                1 for edge in item.required_edges
                if any(kw.lower() in ctx_text for kw in edge.split("->")[0].strip("() ").split("-"))
            )
            graph_recall = edges_found / len(item.required_edges) if item.required_edges else 1.0

        return QuestionResult(
            experiment_id=experiment_id,
            question_id=item.id,
            question_type=item.type,
            strategy=RetrievalStrategy(strategy_name),
            context_precision=ctx_precision,
            graph_recall=graph_recall,
            retrieval_latency_ms=retrieval_ms,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            answer_correctness=answer_correctness,
            generation_latency_ms=generation_ms,
            token_count=token_count,
            error_type=error_type,
            raw_answer=answer,
            raw_context=context,
        )

    def _print_results_summary(self, experiment_id: str, results: list[QuestionResult]):
        """Print a structured summary of evaluation results."""
        logger.info(f"\n{'='*70}")
        logger.info(f"EVALUATION RESULTS — {experiment_id}")
        logger.info(f"{'='*70}")

        # Group by strategy
        strategies = set(r.strategy.value for r in results)
        for strategy in sorted(strategies):
            s_results = [r for r in results if r.strategy.value == strategy]

            # Calculate averages (skip None values)
            def avg(vals):
                valid = [v for v in vals if v is not None]
                return sum(valid) / len(valid) if valid else None

            avg_faith = avg([r.faithfulness for r in s_results])
            avg_cp = avg([r.context_precision for r in s_results])
            avg_ar = avg([r.answer_relevance for r in s_results])
            avg_ac = avg([r.answer_correctness for r in s_results])
            avg_ret_ms = avg([r.retrieval_latency_ms for r in s_results])
            avg_gen_ms = avg([r.generation_latency_ms for r in s_results])
            total_tokens = sum(r.token_count for r in s_results)

            logger.info(f"\n  Strategy: {strategy.upper()}")
            logger.info(f"    Faithfulness:      {avg_faith:.2f}" if avg_faith else "    Faithfulness:      N/A")
            logger.info(f"    Context Precision: {avg_cp:.2f}" if avg_cp else "    Context Precision: N/A")
            logger.info(f"    Answer Relevance:  {avg_ar:.2f}" if avg_ar else "    Answer Relevance:  N/A")
            logger.info(f"    Correctness:       {avg_ac:.2f}" if avg_ac else "    Correctness:       N/A")
            logger.info(f"    Avg Retrieval:     {avg_ret_ms:.0f}ms" if avg_ret_ms else "    Avg Retrieval:     N/A")
            logger.info(f"    Avg Generation:    {avg_gen_ms:.0f}ms" if avg_gen_ms else "    Avg Generation:    N/A")
            logger.info(f"    Total Tokens:      {total_tokens}")

            # Error breakdown
            errors = [r.error_type for r in s_results if r.error_type != EvalErrorType.NONE]
            if errors:
                logger.info(f"    Errors:")
                for et in set(errors):
                    count = errors.count(et)
                    logger.info(f"      - {et.value}: {count}")

        # Breakdown by question type
        logger.info(f"\n  {'─'*50}")
        logger.info(f"  BREAKDOWN BY QUESTION TYPE:")
        q_types = set(r.question_type.value for r in results)
        for qt in sorted(q_types):
            qt_results = [r for r in results if r.question_type.value == qt]
            for strategy in sorted(strategies):
                s_qt_results = [r for r in qt_results if r.strategy.value == strategy]
                ac_vals = [r.answer_correctness for r in s_qt_results if r.answer_correctness is not None]
                avg_ac = sum(ac_vals) / len(ac_vals) if ac_vals else None
                score_str = f"{avg_ac:.2f}" if avg_ac is not None else "N/A"
                logger.info(f"    {qt:15s} | {strategy:7s} | correctness={score_str}")

        logger.info(f"{'='*70}")


if __name__ == "__main__":
    harness = EvaluationHarness()
    # Start with 3 questions to test (respecting API quota)
    harness.run(max_questions=3)
