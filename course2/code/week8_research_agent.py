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
# ## Course 2, Week 8: Capstone — Autonomous Research Agent
# This capstone integrates planning, parallel research, synthesis, contradiction detection,
# and human-in-the-loop (HITL) checkpoints into a single autonomous research pipeline.
# By the end you will have a system that takes a research question and produces a cited,
# contradiction-checked report saved to disk.

# %% [markdown]
# ## 1. Setup
# Import all dependencies, define model constants, and configure logging.

# %%
import os
import asyncio
import json
import logging
import time
import math
import urllib.request
import urllib.parse
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
from mistralai import Mistral, AsyncMistral
from mistralai.models import SDKError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("research_agent")

load_dotenv()

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")

MODEL_LARGE   = "mistral-large-latest"
MODEL_SMALL   = "mistral-small-latest"
MODEL_EMBED   = "mistral-embed"

print("Setup complete. Models:", MODEL_LARGE, MODEL_SMALL, MODEL_EMBED)

# %% [markdown]
# ## 2. Planner Agent
# The PlannerAgent decomposes an open-ended research question into a structured
# ResearchPlan with sub-questions, constraints, and estimated sources.
# A human-approval checkpoint lets the operator review the plan before execution.

# %%
PLANNER_PROMPT = """You are a rigorous research planner. Given a research question,
decompose it into 3-5 targeted sub-questions that together fully answer the original
question. Respond ONLY with valid JSON matching this schema:
{
  "question": "<original question>",
  "sub_questions": ["<sub-q 1>", "<sub-q 2>", ...],
  "constraints": "<scope limits, e.g. 'focus on enterprise context, post-2020'>",
  "estimated_sources": <integer 5-15>
}"""


@dataclass
class ResearchPlan:
    """Structured decomposition of a research question produced by the PlannerAgent."""

    question: str
    sub_questions: list
    constraints: str
    estimated_sources: int

    def print_summary(self) -> None:
        """Print a human-readable plan summary to stdout."""
        print("\n" + "=" * 60)
        print(f"RESEARCH PLAN")
        print("=" * 60)
        print(f"Question : {self.question}")
        print(f"Constraints: {self.constraints}")
        print(f"Est. sources: {self.estimated_sources}")
        print("Sub-questions:")
        for i, sq in enumerate(self.sub_questions, 1):
            print(f"  {i}. {sq}")
        print("=" * 60)


class PlannerAgent:
    """Uses Mistral to decompose a research question into a ResearchPlan."""

    def __init__(self, api_key: str = MISTRAL_API_KEY):
        """Initialize the planner with a Mistral client."""
        self.client = Mistral(api_key=api_key)

    def decompose(self, question: str) -> ResearchPlan:
        """Call Mistral to produce a ResearchPlan from a research question."""
        try:
            response = self.client.chat.complete(
                model=MODEL_LARGE,
                messages=[
                    {"role": "system", "content": PLANNER_PROMPT},
                    {"role": "user", "content": question},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)
        except (SDKError, json.JSONDecodeError) as exc:
            logger.error("PlannerAgent.decompose failed: %s", exc)
            raise

        plan = ResearchPlan(
            question=data.get("question", question),
            sub_questions=data.get("sub_questions", []),
            constraints=data.get("constraints", ""),
            estimated_sources=int(data.get("estimated_sources", 8)),
        )
        self.validate_plan(plan)
        return plan

    @staticmethod
    def validate_plan(plan: ResearchPlan) -> None:
        """Assert that the plan has at least one sub-question."""
        assert len(plan.sub_questions) >= 1, "Plan must have at least one sub-question"
        assert plan.question, "Plan must include the original question"
        logger.info("Plan validated: %d sub-questions", len(plan.sub_questions))

    @staticmethod
    def human_approval(plan: ResearchPlan) -> bool:
        """Print the plan and ask the operator for approval (HITL checkpoint)."""
        plan.print_summary()
        answer = input("\nApprove this research plan? [Y/n]: ").strip().lower()
        approved = answer in ("", "y", "yes")
        print("Plan APPROVED." if approved else "Plan REJECTED.")
        return approved


# Quick smoke-test (no API call)
_dummy_plan = ResearchPlan(
    question="test", sub_questions=["sq1"], constraints="none", estimated_sources=5
)
PlannerAgent.validate_plan(_dummy_plan)
print("PlannerAgent class ready.")

# %% [markdown]
# ## 3. Researcher Agent
# The ResearcherAgent executes each sub-question using lightweight search tools
# (Wikipedia REST API and arXiv API) and returns structured Evidence objects.
# Sub-questions are researched in parallel with `asyncio.gather`.

# %%
@dataclass
class Evidence:
    """Research findings for a single sub-question."""

    question: str
    facts: list
    sources: list
    confidence: float  # 0.0 – 1.0


def web_search(query: str) -> dict:
    """Simulate a web search by returning a stub result (replace with real API)."""
    return {
        "source": f"web://search?q={urllib.parse.quote(query)}",
        "snippet": f"[Web search result for: {query}] — integrate a real search API for production.",
    }


def wikipedia_search(query: str) -> dict:
    """Fetch the first paragraph of a Wikipedia article via the REST API."""
    try:
        encoded = urllib.parse.quote(query.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        return {
            "source": data.get("content_urls", {}).get("desktop", {}).get("page", url),
            "snippet": data.get("extract", "")[:600],
        }
    except Exception as exc:
        logger.warning("wikipedia_search(%s) failed: %s", query, exc)
        return {"source": "wikipedia", "snippet": ""}


def arxiv_search(query: str, max_results: int = 2) -> list:
    """Search arXiv and return title + abstract snippets for top results."""
    try:
        encoded = urllib.parse.quote(query)
        url = (
            f"https://export.arxiv.org/api/query"
            f"?search_query=all:{encoded}&max_results={max_results}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml = resp.read().decode("utf-8")

        results = []
        import re
        entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)
        for entry in entries[:max_results]:
            title_m = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            summary_m = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            link_m = re.search(r'<id>(.*?)</id>', entry)
            results.append({
                "source": link_m.group(1).strip() if link_m else "arxiv",
                "snippet": (
                    (title_m.group(1).strip() if title_m else "") + " — " +
                    (summary_m.group(1).strip()[:400] if summary_m else "")
                ),
            })
        return results
    except Exception as exc:
        logger.warning("arxiv_search(%s) failed: %s", query, exc)
        return []


class ResearcherAgent:
    """Gathers evidence for sub-questions using search tools and Mistral synthesis."""

    RESEARCHER_PROMPT = (
        "You are a precise research assistant. Given search snippets, extract 3-5 "
        "specific, verifiable facts that answer the question. Respond with JSON: "
        '{"facts": ["fact 1", "fact 2", ...], "confidence": <0.0-1.0>}'
    )

    def __init__(self, api_key: str = MISTRAL_API_KEY):
        """Initialize with an async Mistral client for parallel calls."""
        self.async_client = AsyncMistral(api_key=api_key)

    async def research_sub_question(self, question: str) -> Evidence:
        """Research one sub-question; return Evidence with facts and sources."""
        snippets = []
        sources = []

        wiki = wikipedia_search(question)
        if wiki["snippet"]:
            snippets.append(wiki["snippet"])
            sources.append(wiki["source"])

        for item in arxiv_search(question, max_results=2):
            if item["snippet"]:
                snippets.append(item["snippet"])
                sources.append(item["source"])

        web = web_search(question)
        snippets.append(web["snippet"])
        sources.append(web["source"])

        context = "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets))
        prompt = f"Question: {question}\n\nSources:\n{context}"

        try:
            response = await self.async_client.chat.complete_async(
                model=MODEL_SMALL,
                messages=[
                    {"role": "system", "content": self.RESEARCHER_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)
            facts = data.get("facts", [])
            confidence = float(data.get("confidence", 0.7))
        except (SDKError, json.JSONDecodeError) as exc:
            logger.error("ResearcherAgent failed for '%s': %s", question, exc)
            facts, confidence = [], 0.0

        facts = self.deduplicate_facts(facts)
        return Evidence(question=question, facts=facts, sources=sources, confidence=confidence)

    @staticmethod
    def deduplicate_facts(facts: list) -> list:
        """Remove duplicate or near-duplicate facts (exact-match deduplication)."""
        seen = set()
        unique = []
        for f in facts:
            key = f.strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(f)
        return unique

    async def research_all(self, sub_questions: list) -> list:
        """Research all sub-questions in parallel; return list of Evidence."""
        tasks = [self.research_sub_question(q) for q in sub_questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        evidence_list = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("research task raised: %s", r)
            else:
                evidence_list.append(r)
        return evidence_list


print("ResearcherAgent class ready.")

# %% [markdown]
# ## 4. Writer Agent
# The WriterAgent synthesizes all Evidence objects into a structured Report containing
# an executive summary, detailed findings per sub-question, a sources list, and any
# contradictions flagged by the ContradictionDetector.

# %%
WRITER_PROMPT = """You are an expert research writer. Given a research plan and evidence,
produce a comprehensive report. Respond with JSON matching:
{
  "executive_summary": "<2-3 sentence overview>",
  "findings": [
    {"sub_question": "<question>", "finding": "<detailed paragraph>", "citations": [<source indices>]}
  ],
  "contradictions_noted": "<brief note on any conflicting evidence, or 'None detected'>",
  "confidence_overall": <0.0-1.0>
}"""


@dataclass
class Report:
    """Final research report produced by the WriterAgent."""

    question: str
    executive_summary: str
    findings: list          # list of dicts: {sub_question, finding, citations}
    sources: list           # flat deduplicated source list
    contradictions: list    # list of ContradictionPair descriptions
    confidence_overall: float
    elapsed_seconds: float = 0.0

    def to_markdown(self) -> str:
        """Render the report as a Markdown string."""
        lines = [
            f"# Research Report",
            f"**Question:** {self.question}",
            f"**Confidence:** {self.confidence_overall:.0%}  |  "
            f"**Time:** {self.elapsed_seconds:.1f}s",
            "",
            "## Executive Summary",
            self.executive_summary,
            "",
            "## Findings",
        ]
        for item in self.findings:
            lines.append(f"### {item.get('sub_question', '')}")
            lines.append(item.get("finding", ""))
            cited = item.get("citations", [])
            if cited:
                refs = ", ".join(
                    f"[{c}]" for c in cited if isinstance(c, int) and c <= len(self.sources)
                )
                lines.append(f"*Sources: {refs}*")
            lines.append("")

        lines.append("## Sources")
        for i, src in enumerate(self.sources, 1):
            lines.append(f"{i}. {src}")

        if self.contradictions:
            lines.append("")
            lines.append("## Contradictions Detected")
            for cp in self.contradictions:
                lines.append(f"- **{cp.get('fact_a','')}** vs **{cp.get('fact_b','')}**")
                lines.append(f"  _{cp.get('explanation','')}_")

        return "\n".join(lines)


class WriterAgent:
    """Synthesizes Evidence into a coherent Report using Mistral."""

    def __init__(self, api_key: str = MISTRAL_API_KEY):
        """Initialize the writer with a Mistral client."""
        self.client = Mistral(api_key=api_key)

    def synthesize(
        self,
        plan: ResearchPlan,
        all_evidence: list,
        contradictions: list,
        elapsed: float = 0.0,
    ) -> Report:
        """Produce a Report from the plan and all collected Evidence."""
        flat_sources = []
        for ev in all_evidence:
            for s in ev.sources:
                if s not in flat_sources:
                    flat_sources.append(s)

        evidence_block = json.dumps(
            [asdict(ev) for ev in all_evidence], indent=2
        )[:4000]  # stay within context

        prompt = (
            f"Research question: {plan.question}\n"
            f"Constraints: {plan.constraints}\n\n"
            f"Evidence:\n{evidence_block}"
        )

        try:
            response = self.client.chat.complete(
                model=MODEL_LARGE,
                messages=[
                    {"role": "system", "content": WRITER_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)
        except (SDKError, json.JSONDecodeError) as exc:
            logger.error("WriterAgent.synthesize failed: %s", exc)
            data = {
                "executive_summary": "Synthesis failed.",
                "findings": [],
                "confidence_overall": 0.0,
            }

        return Report(
            question=plan.question,
            executive_summary=data.get("executive_summary", ""),
            findings=data.get("findings", []),
            sources=flat_sources,
            contradictions=[asdict(cp) if hasattr(cp, "__dataclass_fields__") else cp
                            for cp in contradictions],
            confidence_overall=float(data.get("confidence_overall", 0.7)),
            elapsed_seconds=elapsed,
        )


print("WriterAgent class ready.")

# %% [markdown]
# ## 5. Contradiction Detector
# The ContradictionDetector embeds all collected facts with `mistral-embed`, uses
# cosine similarity to find candidate pairs, then calls Mistral to confirm genuine
# contradictions. This prevents flagging semantically similar (not contradictory) facts.

# %%
VERIFY_PROMPT = (
    "Do the following two statements genuinely contradict each other? "
    "Reply with JSON: {\"contradicts\": true/false, \"explanation\": \"<one sentence>\"}\n"
    "Statement A: {fact_a}\nStatement B: {fact_b}"
)


@dataclass
class ContradictionPair:
    """A pair of facts that potentially contradict each other."""

    fact_a: str
    fact_b: str
    source_a: str
    source_b: str
    explanation: str


def _cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class ContradictionDetector:
    """Detects contradicting facts among collected Evidence using embeddings."""

    SIMILARITY_THRESHOLD = 0.80  # candidates must be this similar but not identical

    def __init__(self, api_key: str = MISTRAL_API_KEY):
        """Initialize with a Mistral client for both embeddings and verification."""
        self.client = Mistral(api_key=api_key)

    def _embed_facts(self, facts: list) -> list:
        """Return embedding vectors for a list of fact strings."""
        if not facts:
            return []
        try:
            response = self.client.embeddings.create(
                model=MODEL_EMBED, inputs=facts
            )
            return [item.embedding for item in response.data]
        except SDKError as exc:
            logger.error("Embedding call failed: %s", exc)
            return [[] for _ in facts]

    def _verify_contradiction(self, fact_a: str, fact_b: str) -> Optional[str]:
        """Ask Mistral if two facts genuinely contradict; return explanation or None."""
        prompt = VERIFY_PROMPT.format(fact_a=fact_a, fact_b=fact_b)
        try:
            response = self.client.chat.complete(
                model=MODEL_SMALL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            if data.get("contradicts"):
                return data.get("explanation", "Contradiction confirmed.")
        except (SDKError, json.JSONDecodeError) as exc:
            logger.warning("Contradiction verification failed: %s", exc)
        return None

    def find_contradictions(self, all_evidence: list) -> list:
        """Find and verify contradicting fact pairs across all Evidence objects."""
        # Build flat fact list with source attribution
        flat_facts = []
        flat_sources = []
        for ev in all_evidence:
            src = ev.sources[0] if ev.sources else "unknown"
            for fact in ev.facts:
                flat_facts.append(fact)
                flat_sources.append(src)

        if len(flat_facts) < 2:
            return []

        embeddings = self._embed_facts(flat_facts)
        contradictions = []

        for i in range(len(flat_facts)):
            for j in range(i + 1, len(flat_facts)):
                if not embeddings[i] or not embeddings[j]:
                    continue
                sim = _cosine_similarity(embeddings[i], embeddings[j])
                # High similarity but not identical => potential contradiction
                if 0.60 <= sim <= self.SIMILARITY_THRESHOLD:
                    explanation = self._verify_contradiction(flat_facts[i], flat_facts[j])
                    if explanation:
                        contradictions.append(
                            ContradictionPair(
                                fact_a=flat_facts[i],
                                fact_b=flat_facts[j],
                                source_a=flat_sources[i],
                                source_b=flat_sources[j],
                                explanation=explanation,
                            )
                        )

        logger.info("Contradiction detection complete: %d found", len(contradictions))
        return contradictions


print("ContradictionDetector class ready.")

# %% [markdown]
# ## 6. Full Pipeline Integration
# ResearchSystem wires all agents together into an end-to-end async pipeline with
# a HITL checkpoint, parallel research, contradiction detection, and report saving.

# %%
class ResearchSystem:
    """Orchestrates PlannerAgent, ResearcherAgent, ContradictionDetector, and WriterAgent."""

    def __init__(self, api_key: str = MISTRAL_API_KEY, output_dir: str = "."):
        """Initialize all agents and set the output directory."""
        self.planner     = PlannerAgent(api_key)
        self.researcher  = ResearcherAgent(api_key)
        self.detector    = ContradictionDetector(api_key)
        self.writer      = WriterAgent(api_key)
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def research(self, question: str, skip_approval: bool = False) -> Report:
        """Run the full research pipeline; returns a completed Report."""
        start = time.time()
        logger.info("Starting research pipeline for: %s", question)

        # Step 1: Plan
        print("\n[1/4] Planning research...")
        plan = self.planner.decompose(question)

        # Step 2: HITL checkpoint
        if not skip_approval:
            approved = self.planner.human_approval(plan)
            if not approved:
                raise RuntimeError("Research plan rejected by operator.")
        else:
            plan.print_summary()

        # Step 3: Parallel research
        print(f"\n[2/4] Researching {len(plan.sub_questions)} sub-questions in parallel...")
        all_evidence = await self.researcher.research_all(plan.sub_questions)
        print(f"      Collected evidence from {len(all_evidence)} sub-questions.")

        total_facts = sum(len(ev.facts) for ev in all_evidence)
        print(f"      Total facts gathered: {total_facts}")

        # Step 4: Contradiction detection
        print("\n[3/4] Detecting contradictions...")
        contradictions = self.detector.find_contradictions(all_evidence)
        print(f"      Contradictions found: {len(contradictions)}")

        # Step 5: Synthesize report
        print("\n[4/4] Synthesizing final report...")
        elapsed = time.time() - start
        report = self.writer.synthesize(plan, all_evidence, contradictions, elapsed)
        report.elapsed_seconds = time.time() - start

        print(f"\nPipeline complete in {report.elapsed_seconds:.1f}s")
        return report

    def save_report(self, report: Report, filename: str = "research_report.md") -> Path:
        """Save the report as a Markdown file; return the file path."""
        path = self.output_dir / filename
        path.write_text(report.to_markdown(), encoding="utf-8")
        logger.info("Report saved to %s", path)
        print(f"Report saved: {path}")
        return path


print("ResearchSystem class ready.")

# %% [markdown]
# ## 7. Lab Exercise
# Run the full autonomous research pipeline on an enterprise AI question.
# The system will plan, get approval, research in parallel, detect contradictions,
# synthesize a report, and save it — measuring total time throughout.

# %%
async def run_lab(skip_approval: bool = True) -> Report:
    """
    Lab exercise: research RAG vs fine-tuning for enterprise AI.

    Parameters
    ----------
    skip_approval : bool
        Set False to enable the interactive HITL approval prompt.
    """
    RESEARCH_QUESTION = (
        "What are the main differences between RAG and fine-tuning "
        "for enterprise AI applications?"
    )

    print("=" * 60)
    print("CAPSTONE LAB: Autonomous Research Agent")
    print("=" * 60)
    print(f"Question: {RESEARCH_QUESTION}")

    output_dir = Path("d:/tmp")
    system = ResearchSystem(api_key=MISTRAL_API_KEY, output_dir=str(output_dir))

    start = time.time()
    report = await system.research(RESEARCH_QUESTION, skip_approval=skip_approval)
    total_time = time.time() - start

    # Display report preview
    md = report.to_markdown()
    print("\n" + "=" * 60)
    print("REPORT PREVIEW (first 1500 chars)")
    print("=" * 60)
    print(md[:1500])
    if len(md) > 1500:
        print(f"\n... [{len(md) - 1500} more characters]")

    # Save report
    saved_path = system.save_report(report, "research_report.md")

    # Metrics
    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)
    print(f"Total time       : {total_time:.1f}s")
    print(f"Sub-questions    : {len(report.findings)}")
    print(f"Sources cited    : {len(report.sources)}")
    print(f"Contradictions   : {len(report.contradictions)}")
    print(f"Confidence       : {report.confidence_overall:.0%}")
    print(f"Report saved to  : {saved_path}")

    # Assertions
    assert report.executive_summary, "Report must have an executive summary"
    assert len(report.sources) > 0, "Report must cite at least one source"
    print("\nAll assertions passed.")

    return report


# Run the lab (skip_approval=True avoids blocking in notebook execution)
if __name__ == "__main__":
    asyncio.run(run_lab(skip_approval=True))
else:
    # In Jupyter/IPython
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        lab_report = loop.run_until_complete(run_lab(skip_approval=True))
    except RuntimeError:
        lab_report = asyncio.run(run_lab(skip_approval=True))

# %% [markdown]
# ## Key Takeaways
# - A multi-agent architecture (Planner → Researcher → Detector → Writer) lets each
#   agent specialize, making the system easier to test, swap, and scale independently.
# - Human-in-the-loop checkpoints are cheap to add between pipeline stages and
#   dramatically improve operator trust, especially before costly parallel API calls.
# - `asyncio.gather` enables true parallel API calls across sub-questions, cutting
#   wall-clock time proportionally to the number of sub-questions.
# - Contradiction detection via embeddings + LLM verification is a two-phase filter:
#   cosine similarity finds candidates cheaply, and Mistral confirms semantic conflict.
# - Structured JSON output (response_format=json_object) makes agent-to-agent data
#   handoffs reliable and eliminates fragile regex parsing of free-form text.
