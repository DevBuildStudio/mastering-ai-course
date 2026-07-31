"""Chapter 2: Token budget helper."""

from dataclasses import dataclass


@dataclass
class BudgetConfig:
    max_total_tokens: int
    reserve_completion_tokens: int


def available_prompt_tokens(config: BudgetConfig) -> int:
    remaining = config.max_total_tokens - config.reserve_completion_tokens
    return max(0, remaining)


def fits_budget(prompt_tokens: int, config: BudgetConfig) -> bool:
    return prompt_tokens <= available_prompt_tokens(config)


if __name__ == "__main__":
    cfg = BudgetConfig(max_total_tokens=8000, reserve_completion_tokens=1200)
    print("Available prompt tokens:", available_prompt_tokens(cfg))
    print("Prompt fits budget:", fits_budget(5200, cfg))
