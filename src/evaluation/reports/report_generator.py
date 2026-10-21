"""
Report Generator
GraphRAG v2.1

Generates visual reports from evaluation experiments stored in SQLite.
Produces:
1. Strategy comparison bar charts (correctness, faithfulness per strategy)
2. Question type breakdown (which types does GraphRAG excel at?)
3. Trade-off scatter plot (accuracy vs latency vs cost)
4. Error analysis pie chart (root causes of failures)

Usage:
    python -m src.evaluation.reports.report_generator [experiment_id]
"""
import sys
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.infrastructure.database import MetadataStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "data" / "eval" / "reports"


class ReportGenerator:
    """Generates evaluation reports from SQLite experiment data."""

    def __init__(self):
        self.db = MetadataStore()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def generate(self, experiment_id: str):
        """Generate all report charts for an experiment."""
        results = self.db.get_experiment_results(experiment_id)
        if not results:
            logger.error(f"No results found for experiment: {experiment_id}")
            return

        logger.info(f"Generating reports for {experiment_id} ({len(results)} results)...")

        self._plot_strategy_comparison(results, experiment_id)
        self._plot_question_type_breakdown(results, experiment_id)
        self._plot_tradeoff(results, experiment_id)
        self._plot_error_analysis(results, experiment_id)

        logger.info(f"Reports saved to: {OUTPUT_DIR}")

    def _plot_strategy_comparison(self, results: list[dict], exp_id: str):
        """Bar chart: average metrics per strategy."""
        strategies = sorted(set(r["strategy"] for r in results))
        metrics = ["faithfulness", "context_precision", "answer_relevance", "answer_correctness"]
        metric_labels = ["Faithfulness", "Context\nPrecision", "Answer\nRelevance", "Answer\nCorrectness"]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(metrics))
        width = 0.25
        colors = ["#4A90D9", "#50C878", "#FF6B6B"]

        for i, strategy in enumerate(strategies):
            s_results = [r for r in results if r["strategy"] == strategy]
            avgs = []
            for m in metrics:
                vals = [r[m] for r in s_results if r[m] is not None]
                avgs.append(sum(vals) / len(vals) if vals else 0)
            ax.bar(x + i * width, avgs, width, label=strategy.upper(), color=colors[i % len(colors)])

        ax.set_ylabel("Score (0.0 - 1.0)")
        ax.set_title(f"Strategy Comparison — {exp_id}")
        ax.set_xticks(x + width)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{exp_id}_strategy_comparison.png", dpi=150)
        plt.close()
        logger.info(f"  ✓ Strategy comparison chart")

    def _plot_question_type_breakdown(self, results: list[dict], exp_id: str):
        """Grouped bar chart: correctness per question type per strategy."""
        q_types = sorted(set(r["question_type"] for r in results if r["question_type"]))
        strategies = sorted(set(r["strategy"] for r in results))

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(q_types))
        width = 0.25
        colors = ["#4A90D9", "#50C878", "#FF6B6B"]

        for i, strategy in enumerate(strategies):
            avgs = []
            for qt in q_types:
                vals = [r["answer_correctness"] for r in results
                        if r["strategy"] == strategy and r["question_type"] == qt
                        and r["answer_correctness"] is not None]
                avgs.append(sum(vals) / len(vals) if vals else 0)
            ax.bar(x + i * width, avgs, width, label=strategy.upper(), color=colors[i % len(colors)])

        ax.set_ylabel("Answer Correctness (0.0 - 1.0)")
        ax.set_title(f"Correctness by Question Type — {exp_id}")
        ax.set_xticks(x + width)
        ax.set_xticklabels(q_types)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{exp_id}_question_type_breakdown.png", dpi=150)
        plt.close()
        logger.info(f"  ✓ Question type breakdown chart")

    def _plot_tradeoff(self, results: list[dict], exp_id: str):
        """Scatter plot: Correctness vs Total Latency per strategy.
        This answers: 'Is GraphRAG's accuracy improvement worth the extra time?'
        """
        strategies = sorted(set(r["strategy"] for r in results))
        colors = {"naive": "#4A90D9", "graph": "#50C878", "hybrid": "#FF6B6B"}

        fig, ax = plt.subplots(figsize=(8, 6))

        for strategy in strategies:
            s_results = [r for r in results if r["strategy"] == strategy]
            latencies = [r["retrieval_latency_ms"] + r["generation_latency_ms"] for r in s_results]
            correctness = [r["answer_correctness"] if r["answer_correctness"] is not None else 0
                          for r in s_results]

            ax.scatter(latencies, correctness, label=strategy.upper(),
                      color=colors.get(strategy, "#888"), s=100, alpha=0.7)

        ax.set_xlabel("Total Latency (ms)")
        ax.set_ylabel("Answer Correctness")
        ax.set_title(f"Trade-off: Accuracy vs Speed — {exp_id}")
        ax.set_ylim(-0.05, 1.1)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{exp_id}_tradeoff.png", dpi=150)
        plt.close()
        logger.info(f"  ✓ Trade-off scatter plot")

    def _plot_error_analysis(self, results: list[dict], exp_id: str):
        """Pie chart: distribution of error types across all results."""
        error_counts = {}
        for r in results:
            et = r.get("error_type", "none") or "none"
            error_counts[et] = error_counts.get(et, 0) + 1

        # Only show non-zero error types
        labels = [k for k, v in error_counts.items() if v > 0]
        sizes = [error_counts[k] for k in labels]

        if len(labels) <= 1 and "none" in labels:
            logger.info(f"  ⊘ No errors to plot (all answers correct)")
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        colors_map = {
            "none": "#50C878",
            "retrieval_miss": "#FF6B6B",
            "llm_hallucination": "#FFD700",
            "llm_omission": "#FF8C00",
            "wrong_refusal": "#9370DB",
            "correct_refusal": "#87CEEB",
            "context_truncated": "#DEB887",
            "graph_extraction": "#BC8F8F",
        }
        colors = [colors_map.get(l, "#888") for l in labels]

        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%", startangle=140)
        ax.set_title(f"Error Analysis — {exp_id}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{exp_id}_error_analysis.png", dpi=150)
        plt.close()
        logger.info(f"  ✓ Error analysis pie chart")


if __name__ == "__main__":
    generator = ReportGenerator()
    if len(sys.argv) > 1:
        generator.generate(sys.argv[1])
    else:
        # Generate for most recent experiment
        history = generator.db.conn.execute(
            "SELECT experiment_id FROM experiments ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if history:
            generator.generate(history[0])
        else:
            logger.error("No experiments found. Run the harness first.")
