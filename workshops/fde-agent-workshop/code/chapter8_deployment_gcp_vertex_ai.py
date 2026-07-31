"""Chapter 8: Deployment option scoring."""

from dataclasses import dataclass


@dataclass
class DeploymentOption:
    name: str
    latency_score: int
    reliability_score: int
    cost_score: int


def weighted_score(option: DeploymentOption) -> float:
    # Higher is better for this teaching example.
    return (
        option.latency_score * 0.4
        + option.reliability_score * 0.4
        + option.cost_score * 0.2
    )


if __name__ == "__main__":
    options = [
        DeploymentOption("Low latency support", 9, 7, 5),
        DeploymentOption("High accuracy internal", 6, 9, 4),
        DeploymentOption("Cost efficient triage", 7, 6, 9),
    ]
    ranked = sorted(options, key=weighted_score, reverse=True)
    for opt in ranked:
        print(opt.name, round(weighted_score(opt), 2))
