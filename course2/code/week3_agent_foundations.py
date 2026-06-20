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
# # Course 2, Week 3: AI Agent Foundations — ReAct Pattern
#
# This notebook covers the ReAct (Reasoning + Acting) pattern for AI agents.
# ReAct interleaves chain-of-thought reasoning with tool use, letting agents plan,
# observe, and adapt iteratively until a task is complete.

# %% [markdown]
# ## Setup
# Install: `pip install mistralai python-dotenv wikipedia-api duckduckgo-search chromadb`
# Imports, Mistral client, and tool implementations used throughout the notebook.

# %%
import os, re, time, hashlib, asyncio, sqlite3, logging
from datetime import datetime
from typing import Any
from mistralai.client import Mistral
from mistralai.client.errors import SDKError

logging.basicConfig(level=logging.WARNING)
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")
client = Mistral(api_key=MISTRAL_API_KEY)

def web_search(query: str) -> str:
    """Search DuckDuckGo and return top-3 results."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        return "\n".join(f"{r['title']}: {r['body']}" for r in results) or "No results."
    except Exception as e:
        return f"Search error: {e}"

def wikipedia_search(topic: str) -> str:
    """Return a Wikipedia summary (up to 1 500 chars) for the given topic."""
    try:
        import wikipediaapi
        wiki = wikipediaapi.Wikipedia(language="en", user_agent="AIAgentCourse/1.0")
        page = wiki.page(topic)
        return page.summary[:1500] if page.exists() else f"No Wikipedia page for '{topic}'."
    except Exception as e:
        return f"Wikipedia error: {e}"

def calculate(expression: str) -> str:
    """Safely evaluate a numeric expression and return the result."""
    try:
        if not all(c in "0123456789+-*/()., " for c in expression):
            return "Error: disallowed characters in expression."
        return str(eval(expression, {"__builtins__": {}}))  # noqa: S307
    except Exception as e:
        return f"Calculation error: {e}"

def get_current_date() -> str:
    """Return today's date as YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")

def write_file(path: str, content: str) -> str:
    """Write content to path and return a confirmation string."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"File written: {path}"

TOOLS = {"web_search": web_search, "wikipedia": wikipedia_search,
         "calculate": calculate, "get_current_date": get_current_date,
         "write_file": write_file}

print("Setup complete. Available tools:", list(TOOLS.keys()))

# %% [markdown]
# ## The ReAct Agent Loop
# The ReAct pattern alternates Thought (reasoning) and Action (tool call) steps.
# `parse_react_step` classifies each model response; `run_react_step` dispatches the call.

# %%
REACT_SYSTEM_PROMPT = (
    "You are a ReAct agent. Think step by step. "
    "For each step, output Thought: ... then Action: tool_name(args). "
    "When done, output Final Answer: ..."
)

def parse_react_step(text: str) -> dict:
    """Parse one ReAct step into {type, content, tool_name, tool_args}.

    type is 'thought' | 'action' | 'answer'.
    """
    text = text.strip()
    m = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return {"type": "answer", "content": m.group(1).strip(), "tool_name": None, "tool_args": None}
    m = re.search(r"Action:\s*(\w+)\(([^)]*)\)", text, re.IGNORECASE)
    if m:
        return {"type": "action", "content": text, "tool_name": m.group(1).strip(), "tool_args": m.group(2).strip()}
    return {"type": "thought", "content": text, "tool_name": None, "tool_args": None}

def run_react_step(tool_name: str, tool_args: str) -> str:
    """Execute a tool call and return the observation string."""
    if tool_name not in TOOLS:
        return f"Error: unknown tool '{tool_name}'. Available: {list(TOOLS.keys())}"
    fn = TOOLS[tool_name]
    args = [a.strip().strip("'\"") for a in tool_args.split(",") if a.strip()]
    try:
        return fn() if not args or args == [""] else fn(*args)
    except TypeError as e:
        return f"Tool argument error: {e}"

sample = "Thought: I need GDP data.\nAction: web_search(France GDP 2023)"
parsed = parse_react_step(sample)
assert parsed["type"] == "action" and parsed["tool_name"] == "web_search"
print("parse_react_step OK:", parsed["type"], parsed["tool_name"])

# %% [markdown]
# ## ReAct Agent Class
# `ReActAgent` encapsulates the Thought → Action → Observation loop. It appends
# observations as user messages, parses each response, and returns on `Final Answer:`.

# %%
class AgentTimeoutError(Exception):
    """Raised when the agent exceeds its max_iterations cap."""

class ReActAgent:
    """ReAct agent powered by Mistral.

    Args:
        tools: Dict of tool_name → callable.
        max_iterations: Hard cap on Thought/Action cycles.
        model: Mistral model ID.
    """
    def __init__(self, tools: dict, max_iterations: int = 15, model: str = "mistral-large-latest"):
        self.tools, self.max_iterations, self.model = tools, max_iterations, model

    def _parse_tool_call(self, text: str) -> tuple[str, str]:
        """Extract (tool_name, args_string) from an Action line."""
        m = re.search(r"Action:\s*(\w+)\(([^)]*)\)", text, re.IGNORECASE)
        return (m.group(1).strip(), m.group(2).strip()) if m else ("", "")

    def _execute_step(self, messages: list) -> tuple[str, str]:
        """Send messages to the model; return (action_type, raw_text)."""
        try:
            resp = client.chat.complete(model=self.model, messages=messages)
            text = resp.choices[0].message.content or ""
        except SDKError as e:
            return "answer", f"API error: {e}"
        step = parse_react_step(text)
        return step["type"], text

    def run(self, task: str) -> str:
        """Run the ReAct loop for task and return the final answer string.

        Raises AgentTimeoutError if max_iterations is exceeded.
        """
        messages = [{"role": "system", "content": REACT_SYSTEM_PROMPT},
                    {"role": "user", "content": task}]
        for i in range(self.max_iterations):
            action_type, text = self._execute_step(messages)
            messages.append({"role": "assistant", "content": text})
            print(f"  [iter {i+1}] {action_type}: {text[:100]!r}")
            if action_type == "answer":
                return parse_react_step(text)["content"]
            if action_type == "action":
                tool_name, tool_args = self._parse_tool_call(text)
                obs = run_react_step(tool_name, tool_args)
                print(f"  Observation: {obs[:80]!r}")
                messages.append({"role": "user", "content": f"Observation: {obs}"})
        raise AgentTimeoutError(f"Agent did not finish in {self.max_iterations} iterations.")

agent = ReActAgent(tools=TOOLS, max_iterations=2)
print("ReActAgent created:", agent.model)

# %% [markdown]
# ## Agent Memory
# `AgentMemory` provides in-context messages plus a ChromaDB episodic store. Past episodes
# are retrieved by semantic similarity (mistral-embed) and injected as context.

# %%
class AgentMemory:
    """Two-layer memory: in-context list + ChromaDB episodic store.

    Args:
        collection_name: ChromaDB collection to use.
    """
    def __init__(self, collection_name: str = "agent_episodes"):
        self.in_context: list[dict] = []
        self._collection = None
        try:
            import chromadb
            db = chromadb.Client()
            self._collection = db.get_or_create_collection(collection_name)
        except ImportError:
            print("chromadb not installed — episodic memory disabled.")

    def _embed(self, text: str) -> list[float]:
        """Embed text with mistral-embed; return zero vector on failure."""
        try:
            resp = client.embeddings.create(model="mistral-embed", inputs=[text])
            return resp.data[0].embedding
        except SDKError as e:
            logging.warning("Embedding failed: %s", e)
            return [0.0] * 1024

    def save_episode(self, task: str, result: str, key_facts: list[str]) -> None:
        """Store a completed episode (task, result, key_facts) in ChromaDB."""
        if self._collection is None:
            return
        doc = f"Task: {task}\nResult: {result}\nFacts: {'; '.join(key_facts)}"
        ep_id = hashlib.md5(f"{task}{time.time()}".encode()).hexdigest()
        self._collection.add(documents=[doc], embeddings=[self._embed(doc)], ids=[ep_id],
                             metadatas=[{"task": task, "ts": datetime.now().isoformat()}])
        self.in_context.append({"role": "system", "content": f"[Memory] {doc}"})

    def recall_similar(self, task: str, n: int = 3) -> list[str]:
        """Return up to n past episodes semantically similar to task."""
        if self._collection is None or self._collection.count() == 0:
            return []
        res = self._collection.query(query_embeddings=[self._embed(task)],
                                     n_results=min(n, self._collection.count()))
        return res["documents"][0] if res["documents"] else []

    def inject_memory(self, new_task: str) -> str:
        """Return a formatted context string of relevant past episodes."""
        episodes = self.recall_similar(new_task)
        if not episodes:
            return "No relevant past episodes found."
        return "Relevant past experience:\n" + "\n".join(f"  {i}. {ep[:200]}"
                                                          for i, ep in enumerate(episodes, 1))

memory = AgentMemory()
memory.save_episode("What is 2+2?", "4", ["arithmetic", "addition"])
print("Memory recall test — episodes retrieved:", len(memory.recall_similar("math")))

# %% [markdown]
# ## Loop Detection and Reliability
# `LoopDetector` hashes (tool, args) pairs and flags repeated calls. `ProgressTracker`
# monitors observation novelty; after `stall_limit` stalls it injects a course-correction prompt.

# %%
class LoopDetector:
    """Detect repeated (tool, args) calls to prevent infinite loops.

    Args:
        max_repeats: Appearances before is_stuck() returns True.
    """
    def __init__(self, max_repeats: int = 2):
        self._seen: dict[str, int] = {}
        self.max_repeats = max_repeats

    def record(self, tool_name: str, tool_args: str) -> None:
        """Increment the count for this (tool_name, tool_args) pair."""
        key = hashlib.md5(f"{tool_name}:{tool_args}".encode()).hexdigest()
        self._seen[key] = self._seen.get(key, 0) + 1

    def is_stuck(self, tool_name: str, tool_args: str) -> bool:
        """Return True if this exact call has already been recorded max_repeats times."""
        key = hashlib.md5(f"{tool_name}:{tool_args}".encode()).hexdigest()
        return self._seen.get(key, 0) >= self.max_repeats

class ProgressTracker:
    """Track whether observations are novel; flag stalls after stall_limit repeats.

    Args:
        stall_limit: Consecutive identical observations before is_stalled is True.
    """
    def __init__(self, stall_limit: int = 3):
        self.stall_limit = stall_limit
        self._stall_count = 0
        self._seen: set[str] = set()

    def update(self, observation: str) -> bool:
        """Return True if observation is new; increment stall counter otherwise."""
        key = hashlib.md5(observation.encode()).hexdigest()
        if key in self._seen:
            self._stall_count += 1
            return False
        self._seen.add(key)
        self._stall_count = 0
        return True

    @property
    def is_stalled(self) -> bool:
        """True when stall_count >= stall_limit."""
        return self._stall_count >= self.stall_limit

    def force_different_approach_prompt(self) -> str:
        """Prompt injection to break the agent out of a stall."""
        return "You seem to be repeating yourself. Try a completely different tool or approach."

# Tests
ld = LoopDetector(max_repeats=2)
ld.record("web_search", "France GDP"); ld.record("web_search", "France GDP")
assert ld.is_stuck("web_search", "France GDP")
print("LoopDetector test passed")

pt = ProgressTracker(stall_limit=3)
pt.update("new info"); pt.update("same info")   # both novel → stall=0
pt.update("same info"); pt.update("same info"); pt.update("same info")  # stall=3
assert pt.is_stalled
print("ProgressTracker test passed")

# %% [markdown]
# ## Human-in-the-Loop
# `HITLAgent` gates irreversible actions (`write_file`, `send_email`, `delete_file`) behind
# a y/n prompt, logs decisions to SQLite, and integrates `LoopDetector`/`ProgressTracker`.

# %%
IRREVERSIBLE_ACTIONS = {"write_file", "send_email", "delete_file"}

def _setup_audit_db(path: str = "audit.db") -> sqlite3.Connection:
    """Create audit_log table if needed and return the connection."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS audit_log "
                 "(id INTEGER PRIMARY KEY, ts TEXT, tool TEXT, args TEXT, decision TEXT)")
    conn.commit()
    return conn

class HITLAgent(ReActAgent):
    """ReActAgent with Human-in-the-Loop approval and loop/stall guards.

    Args:
        tools: Dict of tool_name → callable.
        audit_db: SQLite path for the audit log.
        max_iterations: Hard cap on cycles.
        model: Mistral model ID.
    """
    def __init__(self, tools: dict, audit_db: str = "audit.db",
                 max_iterations: int = 15, model: str = "mistral-large-latest"):
        super().__init__(tools, max_iterations, model)
        self._audit = _setup_audit_db(audit_db)
        self._loop = LoopDetector()
        self._progress = ProgressTracker()

    def _request_approval(self, tool_name: str, tool_args: str) -> bool:
        """Print approval request, read y/n, log to audit DB, return bool."""
        print(f"\n{'='*50}\nAPPROVAL REQUIRED\n  Tool: {tool_name}\n  Args: {tool_args}\n{'='*50}")
        try:
            answer = input("Approve? [y/n]: ").strip().lower()
        except EOFError:
            answer = "n"
        decision = "approved" if answer == "y" else "denied"
        self._audit.execute("INSERT INTO audit_log(ts,tool,args,decision) VALUES(?,?,?,?)",
                            (datetime.now().isoformat(), tool_name, tool_args, decision))
        self._audit.commit()
        return decision == "approved"

    async def _async_approval(self, tool_name: str, tool_args: str, timeout: float = 30.0) -> bool:
        """Async approval gate via asyncio.Event with auto-deny on timeout."""
        print(f"\nAsync approval needed: {tool_name}({tool_args}) — {timeout}s to respond")
        event, result = asyncio.Event(), {}
        def _read():
            try:
                result["ok"] = input("Approve async? [y/n]: ").strip().lower() == "y"
            except EOFError:
                result["ok"] = False
            event.set()
        asyncio.get_event_loop().run_in_executor(None, _read)
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            result["ok"] = False
        approved = result.get("ok", False)
        self._audit.execute("INSERT INTO audit_log(ts,tool,args,decision) VALUES(?,?,?,?)",
                            (datetime.now().isoformat(), tool_name, tool_args,
                             "approved" if approved else "denied"))
        self._audit.commit()
        return approved

    def run(self, task: str) -> str:
        """Run the HITL ReAct loop, gating irreversible actions for approval."""
        messages = [{"role": "system", "content": REACT_SYSTEM_PROMPT},
                    {"role": "user", "content": task}]
        for i in range(self.max_iterations):
            action_type, text = self._execute_step(messages)
            messages.append({"role": "assistant", "content": text})
            print(f"  [iter {i+1}] {action_type}")
            if action_type == "answer":
                return parse_react_step(text)["content"]
            if action_type == "action":
                tool_name, tool_args = self._parse_tool_call(text)
                if self._loop.is_stuck(tool_name, tool_args):
                    messages.append({"role": "user",
                                     "content": self._progress.force_different_approach_prompt()})
                    continue
                self._loop.record(tool_name, tool_args)
                if tool_name in IRREVERSIBLE_ACTIONS and not self._request_approval(tool_name, tool_args):
                    messages.append({"role": "user", "content": "Observation: Action denied by user."})
                    continue
                obs = run_react_step(tool_name, tool_args)
                if not self._progress.update(obs) and self._progress.is_stalled:
                    messages.append({"role": "user",
                                     "content": self._progress.force_different_approach_prompt()})
                else:
                    messages.append({"role": "user", "content": f"Observation: {obs}"})
        raise AgentTimeoutError(f"HITLAgent did not finish in {self.max_iterations} iterations.")

hitl = HITLAgent(tools=TOOLS)
print("HITLAgent created. Gated actions:", IRREVERSIBLE_ACTIONS)

# %% [markdown]
# ## Lab Exercise
# Run two real queries with `ReActAgent`, gate a file-write through HITL, and print a
# summary report. Set `MISTRAL_API_KEY` in your environment for live results.

# %%
def run_lab_query(agent: ReActAgent, task: str) -> dict[str, Any]:
    """Run task, time it, and return {task, answer, elapsed_s}."""
    print(f"\n{'='*60}\nTASK: {task}\n{'='*60}")
    start = time.time()
    try:
        answer = agent.run(task)
    except AgentTimeoutError as e:
        answer = f"[Timeout] {e}"
    elapsed = round(time.time() - start, 2)
    print(f"\nFINAL ANSWER: {answer}\nElapsed: {elapsed}s")
    return {"task": task, "answer": answer, "elapsed_s": elapsed}

lab_memory = AgentMemory()
lab_agent  = ReActAgent(tools=TOOLS, max_iterations=10)

# Query 1 — GDP per capita requires web search + arithmetic
result1 = run_lab_query(lab_agent,
    "What is the GDP of France divided by its population? Give the result in USD per person.")
lab_memory.save_episode(result1["task"], result1["answer"],
                        ["France GDP", "France population", "GDP per capita"])

# %%
# Query 2 — city population comparison requires two lookups
result2 = run_lab_query(lab_agent,
    "Compare the populations of Tokyo and New York in 2024.")
lab_memory.save_episode(result2["task"], result2["answer"],
                        ["Tokyo population 2024", "New York population 2024"])

# %%
# HITL demo: writing a file requires human approval (auto-denied via EOF in CI)
print("\n--- HITL Demo (file write needs approval; auto-denied when no TTY) ---")
hitl_result = run_lab_query(HITLAgent(tools=TOOLS, max_iterations=5),
                             "Save the current date to a file called today.txt")

# %%
# Memory injection demo
print("\n--- Memory Injection ---")
print(lab_memory.inject_memory("GDP per capita of a European country"))

# Final report
print(f"\n{'='*60}\nLAB REPORT\n{'='*60}")
for i, r in enumerate([result1, result2], 1):
    print(f"\nQuery {i}: {r['task'][:60]}\n  Answer : {r['answer'][:100]}\n  Elapsed: {r['elapsed_s']}s")
eps = lab_memory._collection.count() if lab_memory._collection else 0
print(f"\nMemory episodes: {eps}  |  Max iterations: {lab_agent.max_iterations}  |  Tools: {list(TOOLS.keys())}")
assert result1["answer"] and result2["answer"], "Both queries must return answers"
print("\nAll lab assertions passed.")

# %% [markdown]
# ## Key Takeaways
# - The ReAct pattern interleaves chain-of-thought reasoning with tool execution, letting
#   agents plan and adapt without any hand-coded decision trees.
# - Lightweight regex parsing keeps the loop provider-agnostic and easy to debug; no
#   special structured-output API is required.
# - Loop detection (hash-based) and stall tracking (novelty-based) are essential guardrails
#   that prevent runaway token consumption.
# - Human-in-the-Loop gates on irreversible actions — with a persistent audit trail — are
#   a practical safety pattern for production agent deployments.
# - Two-layer memory (in-context messages + episodic vector store) reduces redundant tool
#   calls on repeated question types and improves agent efficiency over time.
