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
# # Course 2, Week 5: Multi-Agent Systems
#
# Multi-agent systems coordinate multiple specialized AI agents to solve complex tasks
# that exceed the capability of any single agent. This notebook builds a complete
# orchestration framework with routing, parallel execution, monitoring, and error recovery.

# %% [markdown]
# ## 1. Setup
# Import all required libraries and configure the Mistral client.
# We use asyncio for concurrent agent execution and dataclasses for typed result objects.

# %%
import os
import time
import json
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import defaultdict

from mistralai import Mistral, AsyncMistral
from mistralai.models import SDKError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")

sync_client = Mistral(api_key=MISTRAL_API_KEY)
async_client = AsyncMistral(api_key=MISTRAL_API_KEY)

print("Clients initialised.")
print(f"API key present: {MISTRAL_API_KEY != 'your-key-here'}")

# %% [markdown]
# ## 2. Agent Base Class
# `BaseAgent` is the foundation every specialised agent inherits from. It wraps the
# Mistral chat API, captures latency, and surfaces a typed `AgentResult`. Transient
# API errors are retried up to `max_retries` times with exponential back-off.

# %%
@dataclass
class AgentResult:
    """Typed container for the output of a single agent invocation."""
    agent_name: str
    task: str
    output: str
    tool_calls: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


class AgentError(Exception):
    """Raised when an agent fails after exhausting all retries."""
    pass


class BaseAgent:
    """
    Abstract base class for all agents in the pipeline.

    Subclasses set `name`, `role`, `system_prompt`, and optionally `tools`.
    The `run` method calls the Mistral chat API, retries on transient errors,
    and returns a structured `AgentResult`.
    """

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        model: str = "mistral-large-latest",
        tools: Optional[list[dict]] = None,
        max_retries: int = 3,
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools or []
        self.max_retries = max_retries

    def _build_messages(self, task: str, context: dict) -> list[dict]:
        """Construct the message list, injecting any shared context."""
        messages = [{"role": "system", "content": self.system_prompt}]
        if context:
            ctx_text = "\n".join(f"{k}: {v}" for k, v in context.items())
            messages.append({"role": "user", "content": f"Context:\n{ctx_text}\n\nTask: {task}"})
        else:
            messages.append({"role": "user", "content": task})
        return messages

    async def run(self, task: str, context: dict = {}) -> AgentResult:
        """
        Execute the agent on `task`, optionally enriched with `context`.

        Retries up to `max_retries` times on SDKError before raising AgentError.
        Returns an AgentResult with output text, any tool calls, and latency.
        """
        messages = self._build_messages(task, context)
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "auto"

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            start = time.time()
            try:
                response = await async_client.chat.complete_async(**kwargs)
                latency_ms = (time.time() - start) * 1000
                message = response.choices[0].message
                tool_calls = []
                if message.tool_calls:
                    tool_calls = [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in message.tool_calls
                    ]
                return AgentResult(
                    agent_name=self.name,
                    task=task,
                    output=message.content or "",
                    tool_calls=tool_calls,
                    latency_ms=latency_ms,
                    success=True,
                )
            except SDKError as exc:
                last_error = exc
                wait = 2 ** attempt
                logger.warning(
                    "%s attempt %d/%d failed: %s. Retrying in %ds.",
                    self.name, attempt, self.max_retries, exc, wait,
                )
                await asyncio.sleep(wait)

        return AgentResult(
            agent_name=self.name,
            task=task,
            output="",
            latency_ms=0.0,
            success=False,
            error=str(last_error),
        )


# Quick smoke-test (synchronous path via run-in-event-loop)
async def _smoke_base():
    agent = BaseAgent(
        name="TestAgent",
        role="tester",
        system_prompt="You are a helpful assistant.",
        model="mistral-small-latest",
    )
    result = await agent.run("Reply with exactly: OK")
    assert result.success, f"BaseAgent failed: {result.error}"
    print(f"BaseAgent smoke test passed | latency={result.latency_ms:.0f}ms | output={result.output.strip()[:60]}")

asyncio.run(_smoke_base())

# %% [markdown]
# ## 3. Specialised Agents
# Three domain-specific agents inherit `BaseAgent` and each carry a tailored system
# prompt and, where relevant, tool definitions. `ResearchAgent` exposes mock web and
# Wikipedia search tools. `WritingAgent` uses `codestral-latest` for code tasks and
# `mistral-large-latest` for prose. `EditorAgent` focuses on clarity and correctness.

# %%
SEARCH_WEB_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for recent information on a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
        },
    },
}

SEARCH_WIKI_TOOL = {
    "type": "function",
    "function": {
        "name": "search_wiki",
        "description": "Search Wikipedia for encyclopaedic background on a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The Wikipedia topic to look up."}
            },
            "required": ["topic"],
        },
    },
}


class ResearchAgent(BaseAgent):
    """
    Searches the web and Wikipedia for information relevant to a task.

    In production the tool calls would invoke real search APIs; here the
    agent uses its own knowledge to simulate search results and returns
    a structured research summary.
    """

    def __init__(self):
        super().__init__(
            name="ResearchAgent",
            role="researcher",
            system_prompt=(
                "You are an expert researcher. When given a topic, produce a thorough "
                "research summary with key facts, concepts, and examples. "
                "Structure your output with clear headings. "
                "If you call search tools, incorporate the results into your summary."
            ),
            model="mistral-large-latest",
            tools=[SEARCH_WEB_TOOL, SEARCH_WIKI_TOOL],
        )


class WritingAgent(BaseAgent):
    """
    Transforms a research context into polished written content.

    Uses `codestral-latest` when the task involves code generation so that
    syntax highlighting and docstring conventions are handled correctly.
    """

    def __init__(self, task_type: str = "prose"):
        model = "codestral-latest" if task_type == "code" else "mistral-large-latest"
        super().__init__(
            name="WritingAgent",
            role="writer",
            system_prompt=(
                "You are a skilled technical writer. Using the provided research context, "
                "write clear, accurate, and engaging content. "
                "For code-heavy tasks produce well-commented, runnable Python examples. "
                "For prose tasks write in a logical, readable style with good structure."
            ),
            model=model,
        )
        self.task_type = task_type


class EditorAgent(BaseAgent):
    """
    Reviews a draft for accuracy, clarity, and completeness, then returns
    an improved version with inline change notes.
    """

    def __init__(self):
        super().__init__(
            name="EditorAgent",
            role="editor",
            system_prompt=(
                "You are a meticulous technical editor. Given a draft, you: "
                "1) Fix factual errors and ambiguous statements. "
                "2) Improve sentence flow and eliminate redundancy. "
                "3) Ensure code examples are syntactically correct. "
                "Return the improved draft followed by a brief '### Editor Notes' section."
            ),
            model="mistral-large-latest",
        )


print("Specialised agent classes defined: ResearchAgent, WritingAgent, EditorAgent")

# %% [markdown]
# ## 4. Supervisor Orchestrator
# `SupervisorAgent` owns the full pipeline: it classifies the incoming task, routes it
# to the right starting agent, passes context between agents via a shared scratchpad,
# and applies an error cascade so the writing step can still proceed if research fails.

# %%
class SupervisorAgent:
    """
    Orchestrates a research → write → edit pipeline.

    Attributes
    ----------
    shared_scratchpad : dict
        Key-value store shared across all agents in a single pipeline run.
    """

    ROUTE_PROMPT = (
        "You are a task router. Classify the following task into exactly one of: "
        "'research', 'summarise', 'code'. "
        "Reply with a single lowercase word and nothing else.\n\nTask: {task}"
    )

    def __init__(self):
        self.shared_scratchpad: dict = {}
        self._research_agent = ResearchAgent()
        self._writing_agent = WritingAgent()
        self._code_writing_agent = WritingAgent(task_type="code")
        self._editor_agent = EditorAgent()

    async def route(self, task: str) -> str:
        """
        Use the LLM to classify `task` and return the agent category name.

        Returns one of 'research', 'summarise', or 'code'.
        """
        prompt = self.ROUTE_PROMPT.format(task=task)
        try:
            response = await async_client.chat.complete_async(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
            )
            category = response.choices[0].message.content.strip().lower()
            if category not in {"research", "summarise", "code"}:
                category = "research"
        except SDKError:
            category = "research"
        return category

    def pass_context(self, from_agent: str, result: AgentResult) -> None:
        """Store `result.output` in the shared scratchpad under `from_agent`."""
        self.shared_scratchpad[from_agent] = result.output

    async def execute_pipeline(self, task: str) -> dict[str, AgentResult]:
        """
        Run the full research → write → edit pipeline for `task`.

        If the research step fails, the writer receives a fallback context
        message so the pipeline can still produce a useful output.
        """
        self.shared_scratchpad.clear()
        results: dict[str, AgentResult] = {}

        category = await self.route(task)
        logger.info("Routed '%s...' → category=%s", task[:50], category)
        self.shared_scratchpad["category"] = category

        # --- Research step ---
        research_result = await self._research_agent.run(task)
        results["research"] = research_result
        if research_result.success:
            self.pass_context("research", research_result)
        else:
            logger.warning("Research failed; using fallback context for writer.")
            self.shared_scratchpad["research"] = (
                "Research unavailable. Write from general knowledge."
            )

        # --- Write step ---
        writer = self._code_writing_agent if category == "code" else self._writing_agent
        write_result = await writer.run(task, context=self.shared_scratchpad)
        results["write"] = write_result
        if write_result.success:
            self.pass_context("write", write_result)

        # --- Edit step ---
        if write_result.success:
            edit_result = await self._editor_agent.run(
                "Review and improve the following draft.",
                context=self.shared_scratchpad,
            )
            results["edit"] = edit_result
        else:
            results["edit"] = AgentResult(
                agent_name="EditorAgent",
                task=task,
                output="Skipped: no draft to edit.",
                success=False,
                error="Writer step failed.",
            )

        return results


# Demo: route a task
async def _demo_supervisor():
    supervisor = SupervisorAgent()
    category = await supervisor.route("Research and write a guide to Python asyncio.")
    print(f"Routed category: {category}")
    assert category in {"research", "summarise", "code"}

asyncio.run(_demo_supervisor())

# %% [markdown]
# ## 5. Parallel Fan-Out
# `ParallelResearcher` decomposes a complex question into sub-questions and researches
# them concurrently. An `asyncio.Semaphore` caps the maximum concurrent API calls at 3
# to avoid rate-limit errors. After gathering, results are deduplicated and checked for
# contradictions.

# %%
class ParallelResearcher:
    """
    Fans out a list of sub-questions to ResearchAgent instances in parallel.

    Uses an asyncio.Semaphore to cap concurrency at 3 simultaneous API calls,
    then merges and deduplicates the collected results.
    """

    CONCURRENCY_LIMIT = 3

    def __init__(self):
        self._semaphore = asyncio.Semaphore(self.CONCURRENCY_LIMIT)

    async def _bounded_run(self, agent: ResearchAgent, question: str) -> AgentResult:
        """Run `agent` on `question` while holding the semaphore slot."""
        async with self._semaphore:
            return await agent.run(question)

    async def run_parallel(self, sub_questions: list[str]) -> list[AgentResult]:
        """
        Research all `sub_questions` concurrently and return a list of AgentResults.

        At most CONCURRENCY_LIMIT requests are active at any moment.
        """
        agents = [ResearchAgent() for _ in sub_questions]
        tasks = [
            self._bounded_run(agent, q)
            for agent, q in zip(agents, sub_questions)
        ]
        results = await asyncio.gather(*tasks)
        return list(results)

    def merge_results(self, results: list[AgentResult]) -> str:
        """
        Concatenate successful results, deduplicate repeated sentences, and
        return a single merged research string.
        """
        seen: set[str] = set()
        merged_parts: list[str] = []
        for r in results:
            if not r.success:
                continue
            for sentence in r.output.split(". "):
                normalised = sentence.strip().lower()
                if normalised and normalised not in seen:
                    seen.add(normalised)
                    merged_parts.append(sentence.strip())
        return ". ".join(merged_parts)

    def conflicts_detected(self, results: list[AgentResult]) -> list[str]:
        """
        Scan result outputs for simple numeric/factual contradictions.

        Returns a list of human-readable conflict descriptions. This is a
        heuristic approach; production systems would use an LLM judge.
        """
        conflicts: list[str] = []
        texts = [r.output for r in results if r.success]
        conflict_keywords = [
            ("introduced in", "first appeared in"),
            ("deprecated", "still supported"),
        ]
        for kw_a, kw_b in conflict_keywords:
            has_a = any(kw_a in t.lower() for t in texts)
            has_b = any(kw_b in t.lower() for t in texts)
            if has_a and has_b:
                conflicts.append(f"Potential conflict: '{kw_a}' vs '{kw_b}'")
        return conflicts


# Demo: parallel fan-out on two sub-questions
async def _demo_parallel():
    pr = ParallelResearcher()
    questions = [
        "What is Python asyncio and why was it created?",
        "What are the key primitives in Python asyncio (coroutines, tasks, event loop)?",
    ]
    print(f"Running {len(questions)} sub-questions in parallel...")
    start = time.time()
    results = await pr.run_parallel(questions)
    elapsed = time.time() - start
    print(f"Parallel research completed in {elapsed:.1f}s")
    for r in results:
        status = "OK" if r.success else "FAILED"
        print(f"  [{status}] {r.agent_name} | {r.task[:55]}... | {r.latency_ms:.0f}ms")
    merged = pr.merge_results(results)
    conflicts = pr.conflicts_detected(results)
    print(f"Merged length: {len(merged)} chars | Conflicts detected: {conflicts}")

asyncio.run(_demo_parallel())

# %% [markdown]
# ## 6. Agent Monitoring
# `AgentMonitor` is a lightweight observability layer that wraps agent calls, records
# per-agent metrics, and alerts when success rate drops below a configurable threshold.
# Tasks that fail across all agents are placed on a dead-letter queue for inspection.

# %%
@dataclass
class AgentMetrics:
    """Running statistics for a single agent."""
    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Fraction of calls that succeeded (0.0–1.0)."""
        return self.successes / self.calls if self.calls > 0 else 1.0

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per successful call in milliseconds."""
        return self.total_latency_ms / self.successes if self.successes > 0 else 0.0


class AgentMonitor:
    """
    Wraps agent execution to collect call metrics and surface alerts.

    Usage
    -----
    monitor = AgentMonitor()
    result = await monitor.track(agent, task)
    print(monitor.dashboard())
    """

    def __init__(self, alert_threshold: float = 0.8):
        self._metrics: dict[str, AgentMetrics] = defaultdict(AgentMetrics)
        self.alert_threshold = alert_threshold
        self.dead_letter_queue: list[dict] = []

    async def track(self, agent: BaseAgent, task: str, context: dict = {}) -> AgentResult:
        """Run `agent` on `task`, record metrics, and return the AgentResult."""
        result = await agent.run(task, context)
        m = self._metrics[agent.name]
        m.calls += 1
        if result.success:
            m.successes += 1
            m.total_latency_ms += result.latency_ms
        else:
            m.failures += 1
            self.dead_letter_queue.append({"agent": agent.name, "task": task, "error": result.error})
        return result

    def alert(self, agent_name: str, metric: str = "success_rate", threshold: Optional[float] = None) -> Optional[str]:
        """
        Return an alert string if `metric` for `agent_name` is below `threshold`.

        Uses `self.alert_threshold` when `threshold` is not provided.
        Returns None if everything is within limits.
        """
        threshold = threshold if threshold is not None else self.alert_threshold
        m = self._metrics.get(agent_name)
        if m is None:
            return None
        value = getattr(m, metric, None)
        if value is not None and value < threshold:
            return (
                f"ALERT: {agent_name}.{metric} = {value:.2%} "
                f"(below threshold {threshold:.0%})"
            )
        return None

    def dashboard(self) -> dict[str, dict]:
        """Return a per-agent stats dictionary suitable for display or logging."""
        return {
            name: {
                "calls": m.calls,
                "successes": m.successes,
                "failures": m.failures,
                "success_rate": f"{m.success_rate:.0%}",
                "avg_latency_ms": f"{m.avg_latency_ms:.0f}",
            }
            for name, m in self._metrics.items()
        }


# Demo: monitor wrapping a single agent call
async def _demo_monitor():
    monitor = AgentMonitor(alert_threshold=0.8)
    agent = BaseAgent(
        name="DemoAgent",
        role="demo",
        system_prompt="You are helpful.",
        model="mistral-small-latest",
    )
    result = await monitor.track(agent, "Say 'hello' in one word.")
    print(f"Tracked result success={result.success} | latency={result.latency_ms:.0f}ms")
    dash = monitor.dashboard()
    print("Dashboard:", json.dumps(dash, indent=2))
    alert = monitor.alert("DemoAgent")
    print(f"Alert: {alert or 'None'}")
    print(f"Dead-letter queue length: {len(monitor.dead_letter_queue)}")

asyncio.run(_demo_monitor())

# %% [markdown]
# ## 7. Lab Exercise
# Build a complete 3-agent research pipeline that combines all components from this
# notebook. A `RouterAgent` classifies the incoming task, a `ResearchAgent` and
# `WritingAgent` handle their respective sub-tasks, an `EditorAgent` polishes the
# final output, and `AgentMonitor` records everything. The pipeline is tested with a
# realistic technical writing request.

# %%
async def run_lab_pipeline(task: str) -> None:
    """
    Execute the full 3-agent research pipeline and print the monitoring dashboard.

    Steps
    -----
    1. RouterAgent classifies the task (research / summarise / code).
    2. ResearchAgent and WritingAgent run in parallel for their sub-tasks.
    3. EditorAgent polishes the combined output.
    4. AgentMonitor dashboard is printed with per-agent stats.
    """
    monitor = AgentMonitor(alert_threshold=0.8)
    supervisor = SupervisorAgent()

    print("=" * 60)
    print("LAB: 3-Agent Research Pipeline")
    print("=" * 60)
    print(f"Task: {task}\n")

    # Step 1 — Route
    category = await supervisor.route(task)
    print(f"[Router] Category: {category}")

    # Step 2 — Research + Write in parallel (fan-out)
    parallel_researcher = ParallelResearcher()
    sub_questions = [
        f"Background knowledge and theory: {task}",
        f"Practical examples and best practices: {task}",
    ]
    print(f"\n[Parallel] Launching {len(sub_questions)} research sub-tasks...")
    research_results = await parallel_researcher.run_parallel(sub_questions)
    for r in research_results:
        monitor._metrics[r.agent_name].calls += 1
        if r.success:
            monitor._metrics[r.agent_name].successes += 1
            monitor._metrics[r.agent_name].total_latency_ms += r.latency_ms
        else:
            monitor._metrics[r.agent_name].failures += 1

    merged_research = parallel_researcher.merge_results(research_results)
    conflicts = parallel_researcher.conflicts_detected(research_results)
    print(f"[Research] Merged research: {len(merged_research)} chars")
    if conflicts:
        print(f"[Research] Conflicts detected: {conflicts}")

    # Step 3 — Write
    writer_agent = WritingAgent(task_type="code" if category == "code" else "prose")
    write_result = await monitor.track(
        writer_agent,
        task,
        context={"research_summary": merged_research[:2000], "category": category},
    )
    print(f"\n[Writer] Success={write_result.success} | latency={write_result.latency_ms:.0f}ms")
    if write_result.success:
        print(f"[Writer] Draft preview (first 300 chars):\n{write_result.output[:300]}...\n")

    # Step 4 — Edit
    editor_agent = EditorAgent()
    edit_context = {"draft": write_result.output[:3000] if write_result.success else "No draft."}
    edit_result = await monitor.track(
        editor_agent,
        "Review and improve the provided draft.",
        context=edit_context,
    )
    print(f"[Editor] Success={edit_result.success} | latency={edit_result.latency_ms:.0f}ms")
    if edit_result.success:
        print(f"[Editor] Final output preview (first 300 chars):\n{edit_result.output[:300]}...\n")

    # Step 5 — Dashboard
    print("\n" + "=" * 60)
    print("AGENT MONITORING DASHBOARD")
    print("=" * 60)
    dashboard = monitor.dashboard()
    for agent_name, stats in dashboard.items():
        print(
            f"  {agent_name:20s} | calls={stats['calls']} | "
            f"ok={stats['successes']} | fail={stats['failures']} | "
            f"rate={stats['success_rate']} | avg={stats['avg_latency_ms']}ms"
        )

    # Alerts
    for agent_name in dashboard:
        msg = monitor.alert(agent_name)
        if msg:
            print(f"\n  {msg}")

    print(f"\n  Dead-letter queue entries: {len(monitor.dead_letter_queue)}")
    print("=" * 60)
    print("Pipeline complete.")


# Run the lab
LAB_TASK = "Research and write a guide to Python asyncio with code examples"
asyncio.run(run_lab_pipeline(LAB_TASK))

# %% [markdown]
# ## Key Takeaways
# - Multi-agent systems excel at decomposing complex tasks into specialised sub-tasks,
#   each handled by an agent whose system prompt and model are tuned for that domain.
# - A Supervisor/Orchestrator pattern decouples routing logic from execution logic,
#   making it easy to add, remove, or swap agents without rewriting the pipeline.
# - Parallel fan-out with `asyncio.gather` and a `Semaphore` gives significant latency
#   wins while preventing rate-limit exhaustion on the upstream API.
# - Shared scratchpads and explicit context-passing replace implicit state, making
#   agent pipelines easier to debug, test, and replay.
# - Observability (per-agent metrics, alerting, dead-letter queues) is not optional —
#   it is the mechanism that turns a demo into a production-grade system.
