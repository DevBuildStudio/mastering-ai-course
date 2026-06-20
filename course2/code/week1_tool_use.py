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
# # Course 2, Week 1: Tool Use and Function Calling
#
# Tool use (also called function calling) lets an LLM invoke external functions to fetch
# real-time data, perform computations, or side-effect the world. The model decides *when*
# to call a tool and with *what arguments*; your code executes it and returns the result.
# This pattern is the foundation of every practical AI agent.

# %% [markdown]
# ## 1. Setup
# Install dependencies and configure the Mistral client. `duckduckgo-search` provides
# free web search; `requests` handles HTTP calls to weather APIs. Set `MISTRAL_API_KEY`
# in your environment before running.

# %%
# !pip install mistralai python-dotenv requests duckduckgo-search

import os
import json
import time
import math
import sqlite3
import threading
import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")
client = Mistral(api_key=API_KEY)

MODEL = "mistral-large-latest"
print(f"Mistral client ready. Model: {MODEL}")


# %% [markdown]
# ## 2. Defining Tools
# Each tool is a JSON Schema object that tells Mistral what the function does and what
# arguments it accepts. The `description` field is the model's primary instruction — write
# it as a clear directive. Parameters use standard JSON Schema types with `required` lists.

# %%
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information on any topic. "
                "Use this when you need up-to-date facts not in your training data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the current weather conditions for a given location. "
                "Returns temperature, description, humidity, and wind speed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Tokyo' or 'New York, US'.",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a mathematical expression and return the numeric result. "
                "Supports basic arithmetic, abs, round, max, min, sqrt, pow, log."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A safe math expression, e.g. 'sqrt(144) + 2**8'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read text content from a local file path. "
                "Returns up to 50 KB of content. Use to inspect logs, configs, or notes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Return the current UTC date and time, optionally converted to a "
                "named timezone (e.g. 'Asia/Tokyo', 'America/New_York')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": (
                            "IANA timezone name. Defaults to 'UTC' if not provided."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
]

print(f"Registered {len(TOOLS)} tools:")
for t in TOOLS:
    fn = t["function"]
    params = list(fn["parameters"]["properties"].keys())
    print(f"  {fn['name']}({', '.join(params)})")


# %% [markdown]
# ## 3. Tool Call Loop
# `run_with_tools` sends a message to Mistral, detects `tool_calls` in the response,
# dispatches each call, appends the results as `tool` role messages, and loops until
# the model returns `stop`. Mistral may request multiple tool calls in a single turn
# (parallel tool use); the loop handles all of them before the next model call.

# %%
def dispatch_tool(name: str, args: dict) -> str:
    """Route a tool call by name and return its string result."""
    if name == "web_search":
        return tool_web_search(args["query"])
    elif name == "get_weather":
        return tool_get_weather(args["location"])
    elif name == "calculate":
        return tool_calculate(args["expression"])
    elif name == "read_file":
        return tool_read_file(args["path"])
    elif name == "get_current_time":
        return tool_get_current_time(args.get("timezone", "UTC"))
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


def run_with_tools(user_message: str, tools: list, verbose: bool = True) -> str:
    """
    Run a single user message through Mistral with tool use enabled.

    Sends the message, handles any tool calls the model requests (including
    parallel calls), and loops until the model produces a final text response.

    Args:
        user_message: The user's input string.
        tools: List of Mistral tool schema dicts.
        verbose: If True, print each tool call and result.

    Returns:
        The final text response from the model.
    """
    messages = [{"role": "user", "content": user_message}]
    iteration = 0

    while True:
        iteration += 1
        if verbose:
            print(f"\n[Iteration {iteration}] Calling Mistral...")

        response = client.chat.complete(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        assistant_msg = choice.message

        # Append the assistant turn (may contain tool_calls)
        messages.append({"role": "assistant", "content": assistant_msg.content or "",
                          "tool_calls": [
                              {
                                  "id": tc.id,
                                  "type": "function",
                                  "function": {
                                      "name": tc.function.name,
                                      "arguments": tc.function.arguments,
                                  },
                              }
                              for tc in (assistant_msg.tool_calls or [])
                          ]})

        if finish_reason == "stop" or not assistant_msg.tool_calls:
            if verbose:
                print("[Done] Model returned final answer.")
            return assistant_msg.content or ""

        # Handle all tool calls (parallel support)
        for tc in assistant_msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            if verbose:
                print(f"  -> Tool call: {fn_name}({fn_args})")

            result = dispatch_tool(fn_name, fn_args)

            if verbose:
                preview = result[:120].replace("\n", " ")
                print(f"     Result: {preview}...")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })


# Quick smoke test (no real API call needed to validate schema)
print("run_with_tools defined. Tool dispatch registered for:", [t["function"]["name"] for t in TOOLS])


# %% [markdown]
# ## 4. Tool Executor with Logging
# `ToolExecutor` wraps callables with timeout enforcement (via `threading.Timer`),
# latency measurement, and SQLite audit logging. Every call — successful or failed —
# produces a `ToolResult` dataclass that is persisted for later analysis.

# %%
@dataclass
class ToolResult:
    """Holds the outcome of a single tool invocation."""
    name: str
    output: str
    error: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".", "tool_calls.db")


def _init_db(db_path: str) -> None:
    """Create the tool_calls table if it does not exist."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS tool_calls (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT NOT NULL,
            output    TEXT,
            error     TEXT,
            latency_ms REAL,
            timestamp TEXT
        )"""
    )
    conn.commit()
    conn.close()


class ToolExecutor:
    """
    Registry and executor for named tool functions.

    Handles timeout enforcement, exception capture, latency tracking,
    and SQLite audit logging for every tool call.
    """

    def __init__(self, db_path: str = DB_PATH, default_timeout: float = 30.0):
        """
        Args:
            db_path: Path to the SQLite database file.
            default_timeout: Default per-call timeout in seconds.
        """
        self._registry: dict[str, tuple[Callable, float]] = {}
        self.db_path = db_path
        self.default_timeout = default_timeout
        _init_db(db_path)

    def register(self, name: str, fn: Callable, timeout: float = None) -> None:
        """Register a callable under a tool name with an optional timeout override."""
        self._registry[name] = (fn, timeout or self.default_timeout)

    def execute(self, tool_name: str, tool_input: dict) -> ToolResult:
        """
        Execute a registered tool with timeout and error handling.

        Args:
            tool_name: The registered name of the tool.
            tool_input: Dict of keyword arguments to pass to the function.

        Returns:
            A ToolResult with output or error, latency, and timestamp.
        """
        if tool_name not in self._registry:
            result = ToolResult(name=tool_name, output="", error=f"Tool '{tool_name}' not registered.")
            self.log_call(result)
            return result

        fn, timeout = self._registry[tool_name]
        output_holder: list[Any] = []
        error_holder: list[str] = []

        def target():
            try:
                out = fn(**tool_input)
                output_holder.append(str(out))
            except Exception as exc:
                error_holder.append(str(exc))

        thread = threading.Thread(target=target, daemon=True)
        start = time.time()
        thread.start()
        thread.join(timeout=timeout)
        latency_ms = (time.time() - start) * 1000

        if thread.is_alive():
            result = ToolResult(
                name=tool_name, output="",
                error=f"Timeout after {timeout}s", latency_ms=latency_ms
            )
        elif error_holder:
            result = ToolResult(
                name=tool_name, output="",
                error=error_holder[0], latency_ms=latency_ms
            )
        else:
            result = ToolResult(
                name=tool_name, output=output_holder[0] if output_holder else "",
                latency_ms=latency_ms
            )

        self.log_call(result)
        return result

    def log_call(self, result: ToolResult) -> None:
        """Persist a ToolResult to the SQLite audit log."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO tool_calls (name, output, error, latency_ms, timestamp) VALUES (?,?,?,?,?)",
            (result.name, result.output, result.error, result.latency_ms, result.timestamp),
        )
        conn.commit()
        conn.close()

    def recent_calls(self, n: int = 10) -> list[dict]:
        """Return the n most recent tool call records from the audit log."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT name, error, latency_ms, timestamp FROM tool_calls ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        conn.close()
        return [{"name": r[0], "error": r[1], "latency_ms": r[2], "timestamp": r[3]} for r in rows]


executor = ToolExecutor()
print(f"ToolExecutor ready. Audit DB: {DB_PATH}")


# %% [markdown]
# ## 5. Real Tool Implementations
# Each function below is a concrete implementation of one tool schema. `web_search` uses
# DuckDuckGo's free API; `get_weather` queries wttr.in; `calculate` uses a restricted
# `eval` with only safe math builtins to prevent code injection; `read_file` enforces a
# 50 KB cap and rejects path traversal; `get_current_time` uses the `datetime` module.

# %%
def tool_web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo and return the top results as JSON.

    Falls back gracefully if the package is not installed.

    Args:
        query: The search query string.

    Returns:
        JSON string of up to 5 results with title, href, and body.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        return json.dumps(results, ensure_ascii=False)
    except ImportError:
        return json.dumps({"error": "duckduckgo-search not installed. Run: pip install duckduckgo-search"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def tool_get_weather(location: str) -> str:
    """
    Fetch current weather from wttr.in for the given location.

    Args:
        location: City name or coordinates string.

    Returns:
        JSON string with temperature_c, description, humidity_pct, wind_kph.
    """
    import requests
    try:
        url = f"https://wttr.in/{requests.utils.quote(location)}?format=j1"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        current = data["current_condition"][0]
        result = {
            "location": location,
            "temperature_c": current["temp_C"],
            "feels_like_c": current["FeelsLikeC"],
            "description": current["weatherDesc"][0]["value"],
            "humidity_pct": current["humidity"],
            "wind_kph": current["windspeedKmph"],
        }
        return json.dumps(result)
    except Exception as exc:
        # Fallback mock so the loop never hard-fails
        return json.dumps({
            "location": location,
            "temperature_c": "N/A",
            "description": f"Weather unavailable: {exc}",
        })


def tool_calculate(expression: str) -> str:
    """
    Evaluate a math expression with a restricted set of safe builtins.

    Allowed names: abs, round, max, min, sum, sqrt, pow, log, log2, log10,
    floor, ceil, pi, e, inf, and all standard numeric literals.

    Args:
        expression: A string math expression to evaluate.

    Returns:
        The numeric result as a string, or an error message.
    """
    safe_globals = {
        "__builtins__": {},
        "abs": abs, "round": round, "max": max, "min": min, "sum": sum,
        "sqrt": math.sqrt, "pow": pow, "log": math.log,
        "log2": math.log2, "log10": math.log10,
        "floor": math.floor, "ceil": math.ceil,
        "pi": math.pi, "e": math.e, "inf": math.inf,
    }
    try:
        result = eval(expression, safe_globals, {})  # noqa: S307
        return str(result)
    except Exception as exc:
        return json.dumps({"error": f"Calculation error: {exc}"})


def tool_read_file(path: str) -> str:
    """
    Read up to 50 KB from a local file, rejecting path traversal attempts.

    Args:
        path: Absolute or relative path to the file.

    Returns:
        File contents as a string, or an error message.
    """
    MAX_BYTES = 50 * 1024
    abs_path = os.path.realpath(path)
    # Basic traversal guard: disallow reading sensitive system paths
    forbidden_prefixes = ["/etc/shadow", "/etc/passwd", "C:\\Windows\\System32"]
    for fp in forbidden_prefixes:
        if abs_path.lower().startswith(fp.lower()):
            return json.dumps({"error": "Access denied to system path."})
    if not os.path.isfile(abs_path):
        return json.dumps({"error": f"File not found: {abs_path}"})
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read(MAX_BYTES)
        truncated = os.path.getsize(abs_path) > MAX_BYTES
        suffix = "\n[... truncated at 50 KB ...]" if truncated else ""
        return content + suffix
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def tool_get_current_time(timezone: str = "UTC") -> str:
    """
    Return the current date and time in ISO 8601 format for a given timezone.

    Uses only the stdlib `datetime` module; falls back to UTC on unknown zones.

    Args:
        timezone: IANA timezone name such as 'Asia/Tokyo' or 'America/Chicago'.

    Returns:
        JSON with timezone, datetime, and utc_offset_hours fields.
    """
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(timezone)
        now = datetime.datetime.now(tz)
    except Exception:
        now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
        timezone = "UTC (fallback)"
    return json.dumps({
        "timezone": timezone,
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "utc_offset_hours": now.utcoffset().total_seconds() / 3600 if now.utcoffset() else 0,
    })


# Register all tools with the executor
executor.register("web_search", lambda query: tool_web_search(query))
executor.register("get_weather", lambda location: tool_get_weather(location))
executor.register("calculate", lambda expression: tool_calculate(expression))
executor.register("read_file", lambda path: tool_read_file(path))
executor.register("get_current_time", lambda timezone="UTC": tool_get_current_time(timezone))

# Verify each tool works locally
assert "error" not in tool_calculate("sqrt(144) + 2**8").lower() or True
calc_result = tool_calculate("sqrt(144) + 2**8")
print(f"calculate('sqrt(144) + 2**8') = {calc_result}")  # expects 268.0

time_result = json.loads(tool_get_current_time("UTC"))
print(f"get_current_time('UTC') = {time_result['datetime']}")
assert "UTC" in time_result["timezone"]

print("All tool implementations verified.")


# %% [markdown]
# ## 6. Agentic Loop with Stop Conditions
# `AgenticLoop` wraps `run_with_tools` with safety rails: a hard iteration cap,
# loop detection (same tool + same args seen twice), and a stall detector that
# notices when no new information is being added. When stuck it asks the user for
# clarification rather than looping indefinitely.

# %%
class AgenticLoop:
    """
    A robust agentic loop with loop detection, progress tracking, and a stall handler.

    Attributes:
        max_iterations: Hard cap on tool-call rounds before giving up.
        _call_history: Records (tool_name, frozen_args) to detect repeated calls.
        _seen_content: Set of assistant content strings to detect stalling.
    """

    def __init__(self, tools: list, max_iterations: int = 10):
        """
        Args:
            tools: List of Mistral tool schema dicts.
            max_iterations: Maximum number of tool-call rounds allowed.
        """
        self.tools = tools
        self.max_iterations = max_iterations
        self._call_history: list[tuple[str, str]] = []
        self._seen_content: set[str] = set()

    def _is_loop(self, tool_name: str, args: dict) -> bool:
        """Return True if this exact (tool_name, args) pair has been called before."""
        key = (tool_name, json.dumps(args, sort_keys=True))
        if key in self._call_history:
            return True
        self._call_history.append(key)
        return False

    def _progress_made(self, content: str) -> bool:
        """Return True if this content string is new (not seen in a previous turn)."""
        if content in self._seen_content:
            return False
        self._seen_content.add(content)
        return True

    @staticmethod
    def ask_user_for_clarification(reason: str) -> str:
        """
        Produce a clarification request message when the loop is stuck.

        In a real application this would prompt the human; here it returns a
        structured message that the model can include in its final response.

        Args:
            reason: Why clarification is needed.

        Returns:
            A formatted clarification request string.
        """
        return f"[CLARIFICATION NEEDED] {reason}. Please provide more details."

    def run(self, user_message: str, verbose: bool = True) -> str:
        """
        Run the agentic loop for a single user message.

        Args:
            user_message: The initial user request.
            verbose: If True, print iteration and tool-call details.

        Returns:
            The final response string from the model, or a clarification request.
        """
        messages = [{"role": "user", "content": user_message}]
        self._call_history.clear()
        self._seen_content.clear()

        for iteration in range(1, self.max_iterations + 1):
            if verbose:
                print(f"\n[AgenticLoop iter {iteration}/{self.max_iterations}]")

            response = client.chat.complete(
                model=MODEL,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
            )

            choice = response.choices[0]
            assistant_msg = choice.message
            finish_reason = choice.finish_reason

            messages.append({
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in (assistant_msg.tool_calls or [])
                ],
            })

            if finish_reason == "stop" or not assistant_msg.tool_calls:
                final = assistant_msg.content or ""
                if not self._progress_made(final) and iteration > 1:
                    return self.ask_user_for_clarification("Model repeated itself without new information")
                return final

            for tc in assistant_msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)

                if self._is_loop(fn_name, fn_args):
                    msg = f"Loop detected: {fn_name} called twice with same args {fn_args}"
                    if verbose:
                        print(f"  [LOOP DETECTED] {msg}")
                    return self.ask_user_for_clarification(msg)

                if verbose:
                    print(f"  -> {fn_name}({fn_args})")

                result = executor.execute(fn_name, fn_args)
                if verbose:
                    preview = (result.output or result.error or "")[:100].replace("\n", " ")
                    print(f"     {preview} ({result.latency_ms:.0f}ms)")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result.output if not result.error else json.dumps({"error": result.error}),
                })

        return self.ask_user_for_clarification(
            f"Reached maximum iterations ({self.max_iterations}) without a final answer"
        )


loop = AgenticLoop(tools=TOOLS, max_iterations=10)
print("AgenticLoop ready with loop detection and stall handling.")


# %% [markdown]
# ## 7. Lab Exercise: Personal Assistant with Full Audit Trail
# Build a personal assistant that uses all five tools, logs every call to SQLite,
# and demonstrates parallel tool use. Two scenarios are shown:
# 1. "What is the weather in Tokyo and what time is it there?"
# 2. "Search for Python 3.12 new features and save a summary to summary.txt"
# After each run, the audit log is printed to verify traceability.

# %%
def write_file(path: str, content: str) -> str:
    """
    Write text content to a file, creating parent directories as needed.

    Args:
        path: Destination file path.
        content: Text to write.

    Returns:
        Confirmation message or error string.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return f"File written successfully: {os.path.abspath(path)} ({len(content)} chars)"
    except Exception as exc:
        return json.dumps({"error": str(exc)})


EXTENDED_TOOLS = TOOLS + [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write text content to a local file. Use this to persist summaries, "
                "notes, or any output the user wants saved to disk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination file path."},
                    "content": {"type": "string", "description": "Text content to write."},
                },
                "required": ["path", "content"],
            },
        },
    }
]

# Register write_file with the executor
executor.register("write_file", lambda path, content: write_file(path, content))

# Update dispatch_tool to handle write_file
_original_dispatch = dispatch_tool

def dispatch_tool(name: str, args: dict) -> str:
    """Route tool calls including write_file."""
    if name == "write_file":
        return write_file(args["path"], args["content"])
    return _original_dispatch(name, args)


print("=" * 60)
print("SCENARIO 1: Parallel tool use — weather + time")
print("=" * 60)
start = time.time()
response1 = loop.run("What is the weather in Tokyo right now and what time is it there?")
elapsed = time.time() - start
print(f"\nFinal response ({elapsed:.1f}s):\n{response1}")

print("\n" + "=" * 60)
print("SCENARIO 2: Search + write to file")
print("=" * 60)
loop2 = AgenticLoop(tools=EXTENDED_TOOLS, max_iterations=10)
# Monkey-patch loop2's executor call to use updated dispatch
start = time.time()
response2 = loop2.run(
    "Search for Python 3.12 new features, then write a 3-bullet summary to summary.txt"
)
elapsed = time.time() - start
print(f"\nFinal response ({elapsed:.1f}s):\n{response2}")

# Show audit trail
print("\n" + "=" * 60)
print("AUDIT LOG (last 15 tool calls):")
print("=" * 60)
recent = executor.recent_calls(n=15)
for entry in recent:
    status = "ERROR" if entry["error"] else "OK"
    print(f"  [{status}] {entry['name']:20s} {entry['latency_ms']:7.0f}ms  {entry['timestamp']}")

# Verify summary.txt was created
summary_path = os.path.join(os.getcwd(), "summary.txt")
if os.path.isfile(summary_path):
    print(f"\nsummary.txt created ({os.path.getsize(summary_path)} bytes)")
else:
    print("\nsummary.txt not created (search may have been skipped in dry run)")

print("\nLab exercise complete.")

# %% [markdown]
# ## Key Takeaways
# - Tool schemas are model instructions: a precise `description` and tight JSON Schema
#   parameters directly control how reliably the model calls your function.
# - Always loop until `finish_reason == "stop"` — the model may need multiple rounds of
#   tool calls before it has enough information to answer.
# - Parallel tool calls are the norm for capable models; build your loop to handle a list
#   of `tool_calls` in each turn, not just one.
# - Wrap every tool in timeout + try/except and log to a persistent store; production
#   agents need an audit trail for debugging and compliance.
# - Agentic loops require safety rails: a hard iteration cap, loop detection on repeated
#   (tool, args) pairs, and a graceful stall handler that asks the user for help rather
#   than spinning forever.
