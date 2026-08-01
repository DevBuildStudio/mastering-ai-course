# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Week 7: Introduction to Evaluation Systems
#
# Evaluation is the backbone of reliable AI engineering. This notebook covers
# LLM-as-judge patterns, multi-dimensional scoring, async evaluation runners,
# persistent result storage, and visualization — giving you the full toolkit
# to measure, track, and improve any LLM-powered system.

# %% [markdown]
# ## 1. Setup
# We import everything needed: async primitives, JSON/SQLite for storage,
# dataclasses for typed results, and the Mistral client for judge calls.

# %%
import asyncio
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from mistralai import AsyncMistral, Mistral
from mistralai.models import SDKError

matplotlib.rcParams["figure.dpi"] = 100

load_dotenv()

API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")
print("Setup complete.")
print(f"API key present: {API_KEY != 'your-key-here'}")

# %% [markdown]
# ## 2. LLM-as-Judge with Mistral
# A structured judge scores answers on a 1–5 rubric using `mistral-large-latest`
# with JSON mode enabled, returning a numeric score and a reasoning trace.

# %%
JUDGE_PROMPT = """\
You are an expert evaluator assessing an AI assistant's answer.

## Question
{question}

## Reference Answer (may be absent)
{reference}

## Answer to Evaluate
{answer}

## Dimension: {dimension}

## Rubric
5=Excellent, 4=Good, 3=Acceptable, 2=Poor, 1=Unacceptable

Respond with JSON only: {{"score": <1-5>, "reasoning": "<one sentence>"}}
"""


@dataclass
class JudgeResult:
    """Single judge evaluation output."""
    score: int
    reasoning: str
    dimension: str
    latency_ms: float = 0.0


class MistralJudge:
    """LLM-as-judge backed by mistral-large-latest with JSON mode."""

    def __init__(self, api_key: str = API_KEY):
        """Initialise with a Mistral API key."""
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-large-latest"

    def score(self, question: str, answer: str, dimension: str = "CORRECTNESS",
              reference: Optional[str] = None) -> JudgeResult:
        """Score one answer for the given dimension (1-5)."""
        prompt = JUDGE_PROMPT.format(
            question=question, answer=answer, dimension=dimension,
            reference=reference or "Not provided.",
        )
        start = time.time()
        try:
            resp = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            sc, reason = int(data.get("score", 3)), data.get("reasoning", "")
        except SDKError as exc:
            sc, reason = 0, f"API error: {exc}"
        except (json.JSONDecodeError, ValueError) as exc:
            sc, reason = 0, f"Parse error: {exc}"
        return JudgeResult(score=sc, reasoning=reason, dimension=dimension,
                           latency_ms=(time.time() - start) * 1000)


if API_KEY != "your-key-here":
    judge = MistralJudge()
    result = judge.score("What is the capital of France?", "Paris is the capital of France.",
                         "CORRECTNESS", reference="Paris")
    print(f"Score: {result.score}/5 | Latency: {result.latency_ms:.0f} ms")
    print(f"Reasoning: {result.reasoning}")
    assert 1 <= result.score <= 5, "Score must be 1-5"
else:
    print("Skipping live judge test — set MISTRAL_API_KEY to run.")

# %% [markdown]
# ## 3. Multi-Dimension Evaluation
# Five evaluation dimensions are defined as an enum, scored independently,
# and aggregated via configurable weights into a single overall score.

# %%
class EvalDimension(Enum):
    """Supported evaluation dimensions."""
    CORRECTNESS = "CORRECTNESS"
    FAITHFULNESS = "FAITHFULNESS"
    COHERENCE = "COHERENCE"
    SAFETY = "SAFETY"
    TONE = "TONE"


DIMENSION_WEIGHTS: dict[EvalDimension, float] = {
    EvalDimension.CORRECTNESS: 0.35,
    EvalDimension.FAITHFULNESS: 0.25,
    EvalDimension.COHERENCE: 0.20,
    EvalDimension.SAFETY: 0.15,
    EvalDimension.TONE: 0.05,
}


@dataclass
class EvalResult:
    """Complete evaluation result for one question-answer pair."""
    case_id: str
    question: str
    answer: str
    dimension_scores: dict = field(default_factory=dict)
    overall_score: float = 0.0
    total_latency_ms: float = 0.0


def aggregate_scores(dimension_scores: dict[str, JudgeResult]) -> float:
    """Compute weighted average overall score from per-dimension JudgeResults."""
    total_w, weighted_sum = 0.0, 0.0
    for dim, result in dimension_scores.items():
        try:
            enum_dim = EvalDimension(dim)
        except ValueError:
            continue
        if result.score > 0:
            w = DIMENSION_WEIGHTS.get(enum_dim, 0.1)
            weighted_sum += result.score * w
            total_w += w
    return weighted_sum / total_w if total_w > 0 else 0.0


def evaluate_all_dimensions(question: str, answer: str, context: Optional[str] = None,
                            judge: Optional[MistralJudge] = None) -> EvalResult:
    """Score an answer across all five dimensions and return an EvalResult."""
    if judge is None:
        judge = MistralJudge()
    start = time.time()
    dim_scores: dict[str, JudgeResult] = {}
    for dim in EvalDimension:
        ref = context if dim == EvalDimension.FAITHFULNESS else None
        dim_scores[dim.value] = judge.score(question, answer, dim.value, reference=ref)
    overall = aggregate_scores(dim_scores)
    return EvalResult(
        case_id="inline", question=question, answer=answer,
        dimension_scores=dim_scores, overall_score=overall,
        total_latency_ms=(time.time() - start) * 1000,
    )


if API_KEY != "your-key-here":
    er = evaluate_all_dimensions(
        "Explain gradient descent in one sentence.",
        "Gradient descent iteratively adjusts parameters to reduce loss.",
        context="Gradient descent is an optimisation algorithm used in ML.",
    )
    print(f"Overall: {er.overall_score:.2f}/5  |  Latency: {er.total_latency_ms:.0f} ms")
    for dim, jr in er.dimension_scores.items():
        print(f"  {dim}: {jr.score}/5")
else:
    print("Skipping multi-dimension eval — set MISTRAL_API_KEY to run.")

# %% [markdown]
# ## 4. Async Eval Runner
# `EvalRunner` fans out judge calls with `asyncio.gather` and a `Semaphore`
# cap of 10, cutting wall-clock time dramatically versus sequential scoring.

# %%
@dataclass
class RunMetrics:
    """Summary statistics for one evaluation run."""
    model_id: str
    n_cases: int
    mean_overall: float
    per_dimension_means: dict = field(default_factory=dict)
    total_latency_ms: float = 0.0


class EvalRunner:
    """Orchestrates async multi-dimension evaluation of a JSONL dataset."""

    def __init__(self, api_key: str = API_KEY, concurrency: int = 10):
        """Initialise with API key and max concurrent judge requests."""
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(concurrency)

    def load_dataset(self, jsonl_path: str) -> list[dict]:
        """Load evaluation cases from a JSONL file (keys: id, question, answer)."""
        cases = []
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
        print(f"Loaded {len(cases)} cases from {jsonl_path}")
        return cases

    async def _judge_one(self, client: AsyncMistral, case: dict,
                         dimension: str) -> tuple[str, str, JudgeResult]:
        """Score one case/dimension pair, honouring the concurrency semaphore."""
        async with self.semaphore:
            prompt = JUDGE_PROMPT.format(
                question=case.get("question", ""), answer=case.get("answer", ""),
                dimension=dimension,
                reference=case.get("reference", case.get("context", "Not provided.")),
            )
            start = time.time()
            try:
                resp = await client.chat.complete_async(
                    model="mistral-large-latest",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                data = json.loads(resp.choices[0].message.content)
                sc, reason = int(data.get("score", 3)), data.get("reasoning", "")
            except SDKError as exc:
                sc, reason = 0, f"API error: {exc}"
            except (json.JSONDecodeError, ValueError) as exc:
                sc, reason = 0, f"Parse error: {exc}"
            jr = JudgeResult(sc, reason, dimension, (time.time() - start) * 1000)
            return case["id"], dimension, jr

    async def run_eval(self, dataset: list[dict], model_id: str = "mistral-large-latest",
                       dimensions: Optional[list[str]] = None) -> tuple[list[EvalResult], RunMetrics]:
        """Run evaluation async across all cases and dimensions; return results + metrics."""
        if dimensions is None:
            dimensions = [d.value for d in EvalDimension]
        async_client = AsyncMistral(api_key=self.api_key)
        tasks = [self._judge_one(async_client, c, d) for c in dataset for d in dimensions]
        print(f"Scheduling {len(tasks)} judge calls...")
        wall_start = time.time()
        raw = await asyncio.gather(*tasks)
        wall_ms = (time.time() - wall_start) * 1000
        print(f"Complete in {wall_ms:.0f} ms")

        case_map: dict[str, EvalResult] = {
            c["id"]: EvalResult(c["id"], c.get("question", ""), c.get("answer", ""))
            for c in dataset
        }
        for cid, dim, jr in raw:
            case_map[cid].dimension_scores[dim] = jr
            case_map[cid].total_latency_ms += jr.latency_ms
        results = list(case_map.values())
        for er in results:
            er.overall_score = aggregate_scores(er.dimension_scores)

        dim_totals: dict[str, list[float]] = {d: [] for d in dimensions}
        for er in results:
            for dim, jr in er.dimension_scores.items():
                if jr.score > 0:
                    dim_totals[dim].append(jr.score)
        metrics = RunMetrics(
            model_id=model_id, n_cases=len(results),
            mean_overall=float(np.mean([r.overall_score for r in results if r.overall_score > 0])),
            per_dimension_means={d: float(np.mean(s)) for d, s in dim_totals.items() if s},
            total_latency_ms=wall_ms,
        )
        return results, metrics

    def save_results(self, results: list[EvalResult], output_path: str) -> None:
        """Persist EvalResult list to a JSONL file."""
        with open(output_path, "w", encoding="utf-8") as fh:
            for er in results:
                fh.write(json.dumps({
                    "case_id": er.case_id, "question": er.question, "answer": er.answer,
                    "overall_score": er.overall_score,
                    "dimension_scores": {d: {"score": jr.score, "reasoning": jr.reasoning}
                                         for d, jr in er.dimension_scores.items()},
                }) + "\n")
        print(f"Results saved to {output_path}")


print("EvalRunner class defined.")

# %% [markdown]
# ## 5. Results Storage and Analysis
# SQLite gives us a lightweight store for eval history. `EvalDatabase` tracks
# runs and detects performance regressions between model versions.

# %%
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL, model_id TEXT NOT NULL,
    dataset_path TEXT NOT NULL, mean_overall REAL
);
CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES eval_runs(id),
    case_id TEXT NOT NULL, question TEXT, answer TEXT,
    overall_score REAL, scores TEXT
);
"""


class EvalDatabase:
    """Thin SQLite wrapper for evaluation run storage and regression detection."""

    def __init__(self, db_path: str = ":memory:"):
        """Open (or create) the database and apply schema."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def store_run(self, model_id: str, dataset_path: str,
                  results: list[EvalResult], metrics: RunMetrics) -> int:
        """Persist a run and all case results; return the new run ID."""
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        cur = self.conn.execute(
            "INSERT INTO eval_runs (timestamp,model_id,dataset_path,mean_overall) VALUES(?,?,?,?)",
            (ts, model_id, dataset_path, metrics.mean_overall),
        )
        run_id = cur.lastrowid
        self.conn.executemany(
            "INSERT INTO eval_results (run_id,case_id,question,answer,overall_score,scores) VALUES(?,?,?,?,?,?)",
            [(run_id, er.case_id, er.question, er.answer, er.overall_score,
              json.dumps({d: {"score": jr.score, "reasoning": jr.reasoning}
                          for d, jr in er.dimension_scores.items()}))
             for er in results],
        )
        self.conn.commit()
        print(f"Stored run {run_id}: {len(results)} cases, mean={metrics.mean_overall:.2f}")
        return run_id

    def get_run(self, run_id: int) -> dict:
        """Fetch metadata and all result rows for a run."""
        return {
            "meta": self.conn.execute("SELECT * FROM eval_runs WHERE id=?", (run_id,)).fetchone(),
            "results": self.conn.execute("SELECT * FROM eval_results WHERE run_id=?", (run_id,)).fetchall(),
        }

    def compare_runs(self, run_id_1: int, run_id_2: int) -> dict:
        """Compare mean overall scores for two runs; return baseline, candidate, delta."""
        def _mean(rid: int) -> float:
            row = self.conn.execute("SELECT mean_overall FROM eval_runs WHERE id=?", (rid,)).fetchone()
            return row[0] if row else 0.0
        b, c = _mean(run_id_1), _mean(run_id_2)
        return {"baseline": b, "candidate": c, "delta": c - b}

    def regression_detected(self, run_id_1: int, run_id_2: int, threshold: float = 0.1) -> bool:
        """Return True if the candidate drops more than threshold points vs baseline."""
        cmp = self.compare_runs(run_id_1, run_id_2)
        detected = cmp["delta"] < -threshold
        print(f"Regression check: delta={cmp['delta']:.3f} → {'REGRESSION' if detected else 'OK'}")
        return detected


db = EvalDatabase(":memory:")
print("EvalDatabase created — schema OK.")
tables = [r[0] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print(f"Tables: {tables}")

# %% [markdown]
# ## 6. Visualization
# Three panels: a radar chart across all five dimensions, a trend line over
# historical runs, and a per-dimension bar chart. Worst-5 failures are printed.

# %%
def visualize_results(results: list[EvalResult], run_history: Optional[list[dict]] = None,
                      save_path: Optional[str] = None) -> None:
    """Three-panel eval dashboard: radar, trend, bar + worst-5 failure cases."""
    valid = [r for r in results if r.overall_score > 0]
    dims = [d.value for d in EvalDimension]
    dim_means = [
        float(np.mean([r.dimension_scores[d].score for r in valid
                       if d in r.dimension_scores and r.dimension_scores[d].score > 0]) or 0)
        for d in dims
    ]
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle("Evaluation Dashboard", fontsize=14, fontweight="bold")

    # Radar
    ax1 = fig.add_subplot(131, polar=True)
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    vals = dim_means + [dim_means[0]]
    angs = angles + angles[:1]
    ax1.plot(angs, vals, "o-", linewidth=2, color="steelblue")
    ax1.fill(angs, vals, alpha=0.25, color="steelblue")
    ax1.set_thetagrids(np.degrees(angles), dims, fontsize=8)
    ax1.set_ylim(0, 5)
    ax1.set_title("Dimension Radar", pad=15)

    # Trend
    ax2 = fig.add_subplot(132)
    if run_history and len(run_history) > 1:
        ax2.plot([r["run_id"] for r in run_history], [r["mean_overall"] for r in run_history],
                 "o-", color="darkorange", linewidth=2)
        ax2.axhline(y=3.5, color="red", linestyle="--", alpha=0.5, label="3.5 min")
        ax2.legend(fontsize=8)
        ax2.set_xlabel("Run ID")
    else:
        ax2.bar(["Current Run"], [float(np.mean([r.overall_score for r in valid])) if valid else 0],
                color="darkorange")
    ax2.set_ylim(0, 5)
    ax2.set_ylabel("Mean Score")
    ax2.set_title("Score Trend")

    # Bar
    ax3 = fig.add_subplot(133)
    colors = ["#4CAF50" if m >= 4 else "#FF9800" if m >= 3 else "#F44336" for m in dim_means]
    bars = ax3.bar(dims, dim_means, color=colors)
    ax3.set_ylim(0, 5)
    ax3.set_ylabel("Mean Score")
    ax3.set_title("Per-Dimension")
    ax3.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, dim_means):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Dashboard saved to {save_path}")
    plt.show()

    worst = sorted(valid, key=lambda r: r.overall_score)[:5]
    print("\nTop-5 Failure Cases:")
    print("-" * 60)
    for i, er in enumerate(worst, 1):
        print(f"{i}. [Score {er.overall_score:.2f}] {er.question[:70]}")
        for dim, jr in er.dimension_scores.items():
            if jr.score <= 2:
                print(f"   {dim}: {jr.score}/5 — {jr.reasoning[:55]}")


def _make_demo_results() -> list[EvalResult]:
    """Generate synthetic EvalResult objects for visualization demos."""
    rng = np.random.default_rng(42)
    demo = []
    for i in range(12):
        dim_scores = {
            d.value: JudgeResult(int(np.clip(rng.integers(2, 6), 1, 5)), "Demo.", d.value)
            for d in EvalDimension
        }
        er = EvalResult(f"demo_{i}", f"Demo question {i}?", f"Demo answer {i}.")
        er.dimension_scores = dim_scores
        er.overall_score = aggregate_scores(dim_scores)
        demo.append(er)
    return demo


demo_results = _make_demo_results()
visualize_results(demo_results, run_history=[{"run_id": i, "mean_overall": 3.0 + i * 0.15}
                                             for i in range(1, 7)])
print(f"Visualisation complete — {len(demo_results)} demo cases rendered.")

# %% [markdown]
# ## 7. Lab Exercise
# Build an evaluation harness for the Week 5 RAG system: load the seed dataset,
# run async eval on 3 dimensions, persist to SQLite, visualise, and identify
# failure patterns. Add more cases to `_RAW` to reach 30 for full coverage.

# %%
# fmt: off
_RAW = [  # (question, answer, context) — extend to 30 for the full lab
    ("What is retrieval-augmented generation?", "RAG combines a retrieval step with a generative model to ground outputs in external documents.", "RAG is a framework that retrieves relevant passages before generating an answer."),
    ("How does dense retrieval differ from sparse retrieval?", "Dense retrieval uses neural embeddings; sparse retrieval uses keyword overlap like BM25.", "Dense retrieval encodes queries and documents as vectors; sparse uses TF-IDF or BM25."),
    ("What is chunking in RAG?", "Chunking splits documents into smaller pieces so each fits within the model's context window.", "Chunking is the process of dividing long documents into manageable segments."),
    ("Why use embeddings for semantic search?", "Embeddings capture meaning, so semantically similar texts have nearby vectors.", "Embedding models map text to a continuous vector space where meaning is preserved."),
    ("What is a vector database?", "A vector database stores and indexes high-dimensional embeddings for fast similarity search.", "Vector databases are purpose-built to index and query embedding vectors efficiently."),
    ("Explain the role of a reranker in RAG.", "A reranker scores retrieved candidates for relevance to the query and reorders them.", "Rerankers are cross-encoders that jointly encode the query and each passage for precise scoring."),
    ("What is hallucination in LLMs?", "Hallucination occurs when the model generates plausible-sounding but factually incorrect content.", "Hallucination refers to confident but fabricated outputs not grounded in the source material."),
    ("How does context window size affect RAG?", "Larger context windows let you retrieve more passages, reducing information loss.", "Context window limits how many retrieved chunks can be included in a single generation call."),
    ("What is FAISS?", "FAISS is a library for efficient similarity search over dense vector collections.", "Facebook AI Similarity Search (FAISS) provides fast approximate nearest-neighbour search."),
    ("What is the difference between precision and recall?", "Precision is the fraction of retrieved items that are relevant; recall is the fraction of relevant items retrieved.", "Precision measures exactness; recall measures completeness of retrieval."),
]
# fmt: on

LAB_TEST_CASES = [{"id": f"rag_{i:02d}", "question": q, "answer": a, "context": ctx}
                  for i, (q, a, ctx) in enumerate(_RAW)]

assert len(LAB_TEST_CASES) == 10, f"Expected 10 seed cases, got {len(LAB_TEST_CASES)}"
print(f"Prepared {len(LAB_TEST_CASES)} RAG evaluation seed cases.")

LAB_JSONL_PATH = (str(Path(__file__).parent / "rag_eval_cases.jsonl")
                  if "__file__" in dir() else "rag_eval_cases.jsonl")
with open(LAB_JSONL_PATH, "w", encoding="utf-8") as f:
    for case in LAB_TEST_CASES:
        f.write(json.dumps(case) + "\n")
print(f"Test cases written to {LAB_JSONL_PATH}")


async def run_lab_eval() -> tuple[int, list[EvalResult], RunMetrics]:
    """Full lab pipeline: load → async eval → SQLite → visualise → failure patterns."""
    runner = EvalRunner(api_key=API_KEY, concurrency=5)
    dataset = runner.load_dataset(LAB_JSONL_PATH)

    results, metrics = await runner.run_eval(
        dataset=dataset, model_id="mistral-large-latest",
        dimensions=["CORRECTNESS", "FAITHFULNESS", "COHERENCE"],
    )
    print(f"\nRun metrics | model: {metrics.model_id}")
    print(f"  Cases: {metrics.n_cases}  |  Mean overall: {metrics.mean_overall:.2f}/5")
    for dim, mean in metrics.per_dimension_means.items():
        print(f"  {dim}: {mean:.2f}")

    lab_db = EvalDatabase(":memory:")
    run_id = lab_db.store_run("mistral-large-latest", LAB_JSONL_PATH, results, metrics)

    visualize_results(results)
    runner.save_results(results, LAB_JSONL_PATH.replace(".jsonl", "_results.jsonl"))

    low = [r for r in results if 0 < r.overall_score < 3.5]
    print(f"\nFailure patterns ({len(low)} cases below 3.5):")
    _nul = JudgeResult(0, "", "")
    patterns = {
        "Hallucination (low FAITHFULNESS)":
            sum(1 for r in low if r.dimension_scores.get("FAITHFULNESS", _nul).score <= 2),
        "Factual error (low CORRECTNESS)":
            sum(1 for r in low if r.dimension_scores.get("CORRECTNESS", _nul).score <= 2),
        "Incoherent response (low COHERENCE)":
            sum(1 for r in low if r.dimension_scores.get("COHERENCE", _nul).score <= 2),
        "Very low overall (< 2.5)":
            sum(1 for r in low if r.overall_score < 2.5),
        "Context mismatch (faith<=2, correct>=3)":
            sum(1 for r in low
                if r.dimension_scores.get("FAITHFULNESS", _nul).score <= 2
                and r.dimension_scores.get("CORRECTNESS", _nul).score >= 3),
    }
    for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
        print(f"  {pattern}: {count}")
    return run_id, results, metrics


if API_KEY != "your-key-here":
    print("\nRunning lab evaluation (~30 API calls)...")
    lab_run_id, lab_results, lab_metrics = asyncio.run(run_lab_eval())
    print(f"\nLab complete — run ID: {lab_run_id}")
else:
    print("\nLab eval skipped — set MISTRAL_API_KEY to run the full pipeline.")
    visualize_results(_make_demo_results())
    print("Demo visualisation rendered instead.")

# %% [markdown]
# ## Key Takeaways
# - LLM-as-judge patterns use a powerful model with an explicit rubric to
#   score answers at scale, removing the need for exhaustive human labelling.
# - Multi-dimensional evaluation (correctness, faithfulness, coherence, safety,
#   tone) surfaces distinct failure modes that a single score would hide.
# - Async execution with a concurrency semaphore can cut evaluation wall-clock
#   time by an order of magnitude compared to sequential scoring.
# - Persisting results in SQLite enables longitudinal tracking and automated
#   regression detection between model versions or prompt changes.
# - Visualising radar charts, trend lines, and worst-case examples turns raw
#   scores into actionable insights for iterative system improvement.
