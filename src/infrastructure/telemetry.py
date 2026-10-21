"""
Telemetry Module
GraphRAG v2.1 — Infrastructure

Measures and records performance metrics for both pipeline and query operations.
These numbers answer the critical trade-off question:
"Is GraphRAG's accuracy improvement worth the extra latency and cost?"
"""
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StepMetrics:
    """Metrics for a single pipeline or query step."""
    step_name: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    items_processed: int = 0
    errors: int = 0

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost estimate based on Gemini Flash pricing.
        Gemini 3.6 Flash: ~$0.075/1M input, ~$0.30/1M output tokens (free tier: $0)
        We track this for awareness even on free tier.
        """
        input_cost = (self.input_tokens / 1_000_000) * 0.075
        output_cost = (self.output_tokens / 1_000_000) * 0.30
        return input_cost + output_cost

    def stop(self):
        self.end_time = time.perf_counter()

    def to_dict(self) -> dict:
        return {
            "step_name": self.step_name,
            "duration_ms": round(self.duration_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "items_processed": self.items_processed,
            "errors": self.errors,
        }


class PipelineTelemetry:
    """Aggregates metrics across all steps of a pipeline run or evaluation."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.steps: list[StepMetrics] = []
        self._overall_start = time.perf_counter()

    def start_step(self, step_name: str) -> StepMetrics:
        """Begin timing a new step."""
        step = StepMetrics(step_name=step_name)
        self.steps.append(step)
        return step

    @property
    def total_duration_seconds(self) -> float:
        return time.perf_counter() - self._overall_start

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.steps)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.estimated_cost_usd for s in self.steps)

    @property
    def total_errors(self) -> int:
        return sum(s.errors for s in self.steps)

    def summary(self) -> dict:
        """Full telemetry summary for logging and storage."""
        return {
            "run_id": self.run_id,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_errors": self.total_errors,
            "steps": [s.to_dict() for s in self.steps],
        }

    def print_report(self):
        """Pretty-print telemetry to console."""
        print("\n" + "=" * 60)
        print(f"TELEMETRY REPORT — Run: {self.run_id}")
        print("=" * 60)
        for s in self.steps:
            status = "✓" if s.errors == 0 else f"✗ ({s.errors} errors)"
            print(f"  [{status}] {s.step_name}: {s.duration_ms:.0f}ms"
                  f" | {s.items_processed} items"
                  f" | {s.total_tokens} tokens"
                  f" | ${s.estimated_cost_usd:.4f}")
        print("-" * 60)
        print(f"  TOTAL: {self.total_duration_seconds:.1f}s"
              f" | {self.total_tokens} tokens"
              f" | ${self.total_cost_usd:.4f}"
              f" | {self.total_errors} errors")
        print("=" * 60)
