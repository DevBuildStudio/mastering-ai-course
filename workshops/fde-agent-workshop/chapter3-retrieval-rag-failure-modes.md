# Chapter 3: Retrieval and RAG Failure Modes

## Why This Chapter Matters
RAG systems fail silently when retrieval quality drops. Teams often blame the model when recall, chunking, or ranking is the real issue.

## Learning Objectives
- Diagnose low recall and low precision retrieval patterns.
- Improve chunking, indexing, and ranking strategy.
- Add reliability checks before generation.

## Core Concepts
- Embeddings and semantic search basics.
- Chunk strategy and metadata tagging.
- Top-k retrieval and reranking.
- Source attribution and confidence signals.

## Practical Framework
RAG debugging loop:
1. Verify corpus quality.
2. Verify chunk granularity.
3. Measure retrieval hit rate.
4. Validate answer grounding against sources.

## Exercise
Use a small document set and intentionally break retrieval:
- Oversized chunks.
- Missing metadata.
- Low top-k values.
Then recover performance step by step.

## Output
A retrieval diagnosis checklist with recommended remediations.
