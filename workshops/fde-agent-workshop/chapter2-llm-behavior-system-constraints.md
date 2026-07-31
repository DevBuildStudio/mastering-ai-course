# Chapter 2: LLM Behavior and System Constraints

## Why This Chapter Matters
LLM quality is bounded by context, prompt design, and model behavior. Engineering choices around these constraints decide reliability.

## Learning Objectives
- Explain context-window and token-budget trade-offs.
- Recognize common model failure patterns.
- Design prompts and flows for predictable outputs.

## Core Concepts
- Prompt tokens vs completion tokens.
- Sampling behavior and response variance.
- Structured output for downstream automation.
- Latency and cost implications of larger contexts.

## Practical Framework
Constraint-first design:
1. Set max latency target.
2. Set max cost per request.
3. Set required output schema.
4. Tune context and prompting to fit constraints.

## Exercise
Given a fixed token budget:
- Trim prompt sections by value.
- Compare compact prompt vs verbose prompt behavior.
- Evaluate format consistency and factual quality.

## Output
A prompt template with budget guardrails and output schema checks.
