# Chapter 4: Tool Calling and Agent Control Loops

## Why This Chapter Matters
Agents become useful in enterprise settings when they can call trusted tools and APIs safely and consistently.

## Learning Objectives
- Design a planner-tool-validator control loop.
- Define safe tool contracts.
- Handle tool-call failures gracefully.

## Core Concepts
- Tool schema and argument validation.
- Deterministic execution boundaries.
- Retry, timeout, and fallback behavior.
- Confidence-aware answer synthesis.

## Practical Framework
Safe tool integration pattern:
1. Plan intent and required tools.
2. Validate arguments against schema.
3. Execute with timeout and retries.
4. Validate outputs and synthesize response.

## Exercise
Implement one status-check tool and one policy-lookup tool.
- Add schema checks.
- Add timeout handling.
- Add fallback answer when tools fail.

## Output
An agent flow diagram and a validated tool routing map.
