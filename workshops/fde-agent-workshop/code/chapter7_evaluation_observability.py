"""Chapter 7: Eval and observability snapshot."""

from dataclasses import dataclass


@dataclass
class RunMetric:
    latency_ms: int
    grounded: bool
    tool_timeout: bool


def success_rate(runs: list[RunMetric]) -> float:
    if not runs:
        return 0.0
    success = sum(1 for r in runs if r.grounded and not r.tool_timeout)
    return success / len(runs)


def p95_latency(runs: list[RunMetric]) -> int:
    if not runs:
        return 0
    values = sorted(r.latency_ms for r in runs)
    idx = max(0, int(len(values) * 0.95) - 1)
    return values[idx]


if __name__ == "__main__":
    sample = [
        RunMetric(800, True, False),
        RunMetric(1100, True, False),
        RunMetric(1600, False, True),
        RunMetric(900, True, False),
    ]
    print("Success rate:", round(success_rate(sample), 3))
    print("P95 latency:", p95_latency(sample))
