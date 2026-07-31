"""Chapter 10: Capstone scoring template."""

from dataclasses import dataclass


@dataclass
class CapstoneScore:
    architecture: int
    reliability: int
    evaluation: int
    deployment: int
    communication: int


def total_score(score: CapstoneScore) -> int:
    return (
        score.architecture * 25
        + score.reliability * 25
        + score.evaluation * 20
        + score.deployment * 20
        + score.communication * 10
    )


def grade(score_value: int) -> str:
    if score_value >= 850:
        return "Outstanding"
    if score_value >= 700:
        return "Strong"
    if score_value >= 550:
        return "Meets expectations"
    return "Needs improvement"


if __name__ == "__main__":
    sample = CapstoneScore(architecture=8, reliability=8, evaluation=7, deployment=7, communication=9)
    result = total_score(sample)
    print("Score:", result)
    print("Grade:", grade(result))
