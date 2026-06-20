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
# # Course 2, Week 2: The Model Context Protocol (MCP)
#
# MCP (Model Context Protocol) is an open standard that lets AI models discover and call
# tools exposed by external servers — over stdio, SSE, or HTTP transports. This notebook
# walks through building real MCP servers with FastMCP, testing them with the Python
# client, and wiring them to Mistral so the LLM can invoke your tools automatically.

# %% [markdown]
# ## 1. Setup
# Install the `mcp` and `mistralai` packages, then import the core primitives we need
# throughout the notebook.

# %%
# pip install mcp mistralai python-dotenv
# (run once in your environment)

import os
import json
import sqlite3
import asyncio
import tempfile
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")
client = Mistral(api_key=MISTRAL_API_KEY)

print("Mistral client ready.")
print(f"mcp package version: ", end="")
try:
    import mcp
    print(mcp.__version__)
except Exception as e:
    print(f"(import error: {e})")

# %% [markdown]
# ## 2. MCP Server Basics
# `FastMCP` is a high-level wrapper in the `mcp` package that lets you register tools
# with a decorator — similar to FastAPI route decorators. The server exposes tools over
# **stdio transport** (subprocess pipes, great for local use) or **SSE transport**
# (HTTP Server-Sent Events, great for remote/cloud deployments). Tools return plain
# Python values; FastMCP serialises them for you.

# %%
try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.stdio import stdio_server
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    print("FastMCP imported successfully.")
except ImportError as e:
    print(f"FastMCP import error: {e}")
    print("Install with: pip install mcp")

# %%
# Minimal demo server — not run here, just illustrates the pattern.
# Save this block to a file and run `python my_server.py` to start it.

DEMO_SERVER_CODE = '''
from mcp.server.fastmcp import FastMCP

app = FastMCP("my-tools-server")   # name shown to the LLM client

@app.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b

@app.resource("greeting://hello")
def get_greeting() -> str:
    """A static resource that returns a greeting string."""
    return "Hello from MCP!"

# Stdio transport  — connect via subprocess
# app.run()

# SSE transport   — connect via HTTP on port 8000
# app.run(transport="sse", port=8000)
'''

print("=== Demo server skeleton ===")
print(DEMO_SERVER_CODE)

# Transport comparison
print("=== Transport comparison ===")
transports = {
    "stdio": "Subprocess pipes. Zero network config. Ideal for local CLI tools.",
    "sse":   "HTTP + Server-Sent Events. Works across machines / containers.",
}
for name, desc in transports.items():
    print(f"  {name:6s}: {desc}")

# %% [markdown]
# ## 3. Building a File Tools MCP Server
# A practical MCP server that lets an LLM read files, list directories, and search
# inside files — with a safety check that restricts access to an allowlist of
# directories so the model cannot read arbitrary paths on your machine.

# %%
from mcp.server.fastmcp import FastMCP

# Safety: only allow access inside these directories
ALLOWED_DIRS = [
    Path(tempfile.gettempdir()),
    Path.home() / "Documents",
    Path("d:/gith/courses"),
]

def _is_allowed(path: str) -> bool:
    """Return True if *path* resolves inside one of ALLOWED_DIRS."""
    resolved = Path(path).resolve()
    return any(
        resolved == allowed.resolve() or allowed.resolve() in resolved.parents
        for allowed in ALLOWED_DIRS
    )

file_server = FastMCP("file-tools")

@file_server.tool()
def read_file(path: str, max_chars: int = 5000) -> str:
    """Read a text file and return its contents (up to max_chars characters).

    Args:
        path: Absolute or relative path to the file.
        max_chars: Maximum number of characters to return (default 5000).

    Returns:
        File contents as a string, truncated if necessary.
    """
    if not _is_allowed(path):
        return f"Access denied: {path} is outside allowed directories."
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n...[truncated at {max_chars} chars]"
        return text
    except FileNotFoundError:
        return f"Error: file not found — {path}"
    except Exception as exc:
        return f"Error reading file: {exc}"

@file_server.tool()
def list_directory(path: str) -> list:
    """List files and subdirectories inside *path*.

    Args:
        path: Directory path to list.

    Returns:
        List of dicts with keys: name, type ('file'|'dir'), size_bytes.
    """
    if not _is_allowed(path):
        return [{"error": f"Access denied: {path}"}]
    try:
        entries = []
        for item in sorted(Path(path).iterdir()):
            entries.append({
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "size_bytes": item.stat().st_size if item.is_file() else None,
            })
        return entries
    except Exception as exc:
        return [{"error": str(exc)}]

@file_server.tool()
def search_in_file(path: str, query: str) -> list:
    """Search for lines containing *query* (case-insensitive) in a text file.

    Args:
        path: Path to the file to search.
        query: Substring to look for.

    Returns:
        List of dicts with keys: line_number, line_text.
    """
    if not _is_allowed(path):
        return [{"error": f"Access denied: {path}"}]
    try:
        results = []
        with open(path, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, start=1):
                if query.lower() in line.lower():
                    results.append({"line_number": i, "line_text": line.rstrip()})
        return results
    except FileNotFoundError:
        return [{"error": f"File not found: {path}"}]
    except Exception as exc:
        return [{"error": str(exc)}]

# Smoke-test the tools directly (without MCP transport)
_tmp = Path(tempfile.gettempdir()) / "mcp_test.txt"
_tmp.write_text("Hello from MCP!\nLine two contains the word Python.\nLine three.\n")

print("list_directory:", list_directory(tempfile.gettempdir())[:3])
print("read_file:", read_file(str(_tmp)))
print("search_in_file:", search_in_file(str(_tmp), "python"))

# %% [markdown]
# ## 4. Building a Database MCP Server
# An SQLite-backed MCP server that lets the LLM query data safely. Only `SELECT`
# statements are permitted — any attempt to run `INSERT`, `UPDATE`, `DROP`, etc.
# is rejected before it reaches the database engine.

# %%
# Create a demo SQLite database in the temp directory
DB_PATH = Path(tempfile.gettempdir()) / "mcp_demo.db"

def _create_demo_db(db_path: Path) -> None:
    """Populate a demo database with sample tables."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS employees (
            id       INTEGER PRIMARY KEY,
            name     TEXT NOT NULL,
            dept     TEXT NOT NULL,
            salary   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS projects (
            id      INTEGER PRIMARY KEY,
            title   TEXT NOT NULL,
            budget  REAL NOT NULL,
            owner   INTEGER REFERENCES employees(id)
        );
        DELETE FROM employees;
        DELETE FROM projects;
        INSERT INTO employees VALUES (1,'Alice','Engineering',95000);
        INSERT INTO employees VALUES (2,'Bob','Marketing',72000);
        INSERT INTO employees VALUES (3,'Carol','Engineering',102000);
        INSERT INTO projects  VALUES (1,'Alpha',50000,1);
        INSERT INTO projects  VALUES (2,'Beta',30000,2);
        INSERT INTO projects  VALUES (3,'Gamma',80000,3);
    """)
    conn.commit()
    conn.close()

_create_demo_db(DB_PATH)
print(f"Demo database created at {DB_PATH}")

db_server = FastMCP("sqlite-db")

_READ_ONLY_PREFIXES = ("select", "with", "explain", "pragma")

def _is_read_only(sql: str) -> bool:
    """Return True only if *sql* starts with a read-only keyword."""
    first = sql.strip().lower().split()[0] if sql.strip() else ""
    return first in _READ_ONLY_PREFIXES

@db_server.tool()
def execute_query(sql: str) -> list:
    """Execute a read-only SQL query and return rows as a list of dicts.

    Args:
        sql: A SELECT (or EXPLAIN / WITH / PRAGMA) statement.

    Returns:
        List of row dicts, or an error dict if the query is not allowed.
    """
    if not _is_read_only(sql):
        return [{"error": "Only SELECT / read-only queries are permitted."}]
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except sqlite3.Error as exc:
        return [{"error": str(exc)}]

@db_server.tool()
def list_tables() -> list:
    """List all user tables in the database.

    Returns:
        List of table name strings.
    """
    rows = execute_query(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    return [r["name"] for r in rows if "name" in r]

@db_server.tool()
def describe_table(table_name: str) -> dict:
    """Return column metadata for *table_name*.

    Args:
        table_name: Name of the table to describe.

    Returns:
        Dict with 'table' and 'columns' keys; columns is a list of dicts
        with cid, name, type, notnull, dflt_value, pk fields.
    """
    rows = execute_query(f"PRAGMA table_info({table_name})")
    return {"table": table_name, "columns": rows}

# Smoke-test
print("Tables:", list_tables())
print("Describe employees:", describe_table("employees"))
print("Query:", execute_query("SELECT name, salary FROM employees ORDER BY salary DESC"))

# Read-only enforcement
print("Blocked write:", execute_query("DROP TABLE employees"))

# %% [markdown]
# ## 5. MCP Resources
# Resources are read-only data endpoints — think of them as named blobs the LLM can
# fetch without triggering side effects. They support static URIs like
# `config://app-settings` and dynamic URI templates like `schema://tables/{table_name}`.
# The client calls `session.read_resource(uri)` to retrieve them.

# %%
CONFIG = {
    "app_name": "MCP Demo",
    "version": "1.0.0",
    "max_query_rows": 500,
    "allowed_tables": ["employees", "projects"],
    "debug": False,
}

resource_server = FastMCP("resource-demo")

@resource_server.resource("config://app-settings")
def get_config() -> str:
    """Static resource: return the application configuration as JSON.

    Returns:
        JSON string of the CONFIG dictionary.
    """
    return json.dumps(CONFIG, indent=2)

@resource_server.resource("schema://tables/{table_name}")
def get_schema(table_name: str) -> str:
    """Dynamic resource: return the schema for a specific table as JSON.

    Args:
        table_name: Name of the table whose schema is requested.

    Returns:
        JSON string with column metadata, or an error message.
    """
    info = describe_table(table_name)
    if not info.get("columns"):
        return json.dumps({"error": f"Table '{table_name}' not found or empty."})
    return json.dumps(info, indent=2)

# Demonstrate calling resource handlers directly
print("=== Static resource: config://app-settings ===")
print(get_config())

print("\n=== Dynamic resource: schema://tables/employees ===")
print(get_schema("employees"))

print("\n=== Dynamic resource: schema://tables/projects ===")
print(get_schema("projects"))

# %% [markdown]
# ## 6. Testing the MCP Server with a Python Client
# The `mcp` package ships an async client. We launch the server as a subprocess via
# `StdioServerParameters`, open a `ClientSession`, call `list_tools()` to discover what
# the server exposes, then invoke a tool with `call_tool()`. Finally we pass the
# discovered tool schemas to Mistral so the model can decide which tool to call.

# %%
# Write the file-tools server to a temp script so we can launch it as a subprocess.
FILE_SERVER_SCRIPT = Path(tempfile.gettempdir()) / "file_tools_server.py"
FILE_SERVER_SCRIPT.write_text(f"""
import sys, os, tempfile
from pathlib import Path
from mcp.server.fastmcp import FastMCP

ALLOWED_DIRS = [Path(tempfile.gettempdir())]

def _is_allowed(path):
    resolved = Path(path).resolve()
    return any(
        resolved == a.resolve() or a.resolve() in resolved.parents
        for a in ALLOWED_DIRS
    )

server = FastMCP("file-tools")

@server.tool()
def read_file(path: str, max_chars: int = 5000) -> str:
    \"\"\"Read a text file and return its contents.\"\"\"
    if not _is_allowed(path):
        return f"Access denied: {{path}}"
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return text[:max_chars]
    except Exception as exc:
        return f"Error: {{exc}}"

@server.tool()
def list_directory(path: str) -> list:
    \"\"\"List files and subdirectories in path.\"\"\"
    if not _is_allowed(path):
        return [{{"error": f"Access denied: {{path}}"}}]
    try:
        return [
            {{"name": i.name, "type": "dir" if i.is_dir() else "file"}}
            for i in sorted(Path(path).iterdir())
        ]
    except Exception as exc:
        return [{{"error": str(exc)}}]

server.run()
""", encoding="utf-8")

print(f"Server script written to: {FILE_SERVER_SCRIPT}")

# %%
async def test_mcp_client(server_script: Path) -> dict:
    """Launch the file-tools server and test it with the MCP client.

    Args:
        server_script: Path to the Python MCP server script.

    Returns:
        Dict with discovered tool names and a sample call result.
    """
    params = StdioServerParameters(
        command="python",
        args=[str(server_script)],
    )
    results: dict[str, Any] = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover tools
            tools_response = await session.list_tools()
            tool_names = [t.name for t in tools_response.tools]
            results["tools"] = tool_names

            # Call list_directory on the temp dir
            call_result = await session.call_tool(
                "list_directory",
                {"path": tempfile.gettempdir()},
            )
            entries = json.loads(call_result.content[0].text) if call_result.content else []
            results["list_directory_count"] = len(entries)

            # Build Mistral-compatible tool schemas from discovered tools
            mistral_tools = []
            for t in tools_response.tools:
                mistral_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema,
                    },
                })
            results["mistral_tool_schemas"] = mistral_tools
    return results

# Run the async test
start = time.time()
try:
    test_results = asyncio.run(test_mcp_client(FILE_SERVER_SCRIPT))
    elapsed = time.time() - start
    print(f"MCP client test completed in {elapsed:.2f}s")
    print(f"Discovered tools: {test_results['tools']}")
    print(f"Entries in temp dir: {test_results['list_directory_count']}")
    print("Mistral tool schemas (first tool):")
    if test_results["mistral_tool_schemas"]:
        print(json.dumps(test_results["mistral_tool_schemas"][0], indent=2))
except Exception as exc:
    print(f"MCP client test error: {exc}")
    test_results = {"mistral_tool_schemas": []}

# %%
# Connect Mistral to the discovered MCP tools
def ask_mistral_with_mcp_tools(
    question: str,
    mistral_tools: list,
    session_call_tool_fn=None,
) -> str:
    """Send a question to Mistral using MCP-discovered tool schemas.

    Args:
        question: Natural language question for the model.
        mistral_tools: List of Mistral-format tool dicts from MCP discovery.
        session_call_tool_fn: Optional callable(name, args) -> str for real calls.
            If None, the function prints the tool the model would invoke.

    Returns:
        Final assistant response string.
    """
    messages = [{"role": "user", "content": question}]
    start = time.time()
    try:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=messages,
            tools=mistral_tools if mistral_tools else None,
            tool_choice="auto" if mistral_tools else None,
        )
        elapsed = time.time() - start
        msg = response.choices[0].message

        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn = tc.function
                print(f"[{elapsed:.2f}s] Model wants to call: {fn.name}({fn.arguments})")
                if session_call_tool_fn:
                    tool_result = session_call_tool_fn(fn.name, json.loads(fn.arguments))
                    messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                    messages.append({"role": "tool", "content": str(tool_result), "tool_call_id": tc.id})
                    followup = client.chat.complete(model="mistral-large-latest", messages=messages)
                    return followup.choices[0].message.content
            return f"(model invoked tool; no follow-up requested)"
        return msg.content
    except Exception as exc:
        return f"Mistral error: {exc}"

if test_results.get("mistral_tool_schemas"):
    answer = ask_mistral_with_mcp_tools(
        "List the files in my temp directory.",
        test_results["mistral_tool_schemas"],
    )
    print(f"Mistral response: {answer}")
else:
    print("Skipping Mistral call — no tool schemas available (MCP server may not have started).")

# %% [markdown]
# ## 7. Lab Exercise — GitHub-Style MCP Server
# Build a complete MCP server that mimics the GitHub API (using local mock data so no
# token is required). The server exposes four tools: `list_repos`, `get_file_content`,
# `list_issues`, and `search_code`. You then connect an MCP client and ask Mistral to
# answer questions about a repository — the model decides which tools to call.

# %%
# --- Mock data store ---------------------------------------------------
MOCK_REPOS = {
    "alice": [
        {"name": "web-app", "language": "Python", "stars": 42, "open_issues": 3},
        {"name": "data-pipeline", "language": "Python", "stars": 18, "open_issues": 1},
        {"name": "cli-tool", "language": "Go", "stars": 7, "open_issues": 0},
    ],
}

MOCK_FILES: dict[str, dict[str, str]] = {
    "web-app": {
        "README.md": "# web-app\nA Flask web application with REST API.\n\n## Setup\npip install -r requirements.txt\n",
        "app.py": "from flask import Flask\napp = Flask(__name__)\n\n@app.route('/')\ndef index():\n    return 'Hello World'\n",
        "requirements.txt": "flask==3.0.0\ngunicorn==21.2.0\n",
    },
    "data-pipeline": {
        "README.md": "# data-pipeline\nETL pipeline using pandas and SQLAlchemy.\n",
        "pipeline.py": "import pandas as pd\nfrom sqlalchemy import create_engine\n\ndef run_pipeline(source, dest):\n    df = pd.read_csv(source)\n    engine = create_engine(dest)\n    df.to_sql('output', engine, if_exists='replace')\n",
    },
}

MOCK_ISSUES: dict[str, list[dict]] = {
    "web-app": [
        {"id": 1, "title": "500 error on /api/users", "state": "open", "labels": ["bug"]},
        {"id": 2, "title": "Add authentication middleware", "state": "open", "labels": ["enhancement"]},
        {"id": 3, "title": "Update Flask to 3.x", "state": "closed", "labels": ["maintenance"]},
    ],
    "data-pipeline": [
        {"id": 1, "title": "Memory leak with large CSV files", "state": "open", "labels": ["bug"]},
    ],
}

# %%
# --- GitHub-style MCP server ------------------------------------------
github_server = FastMCP("github-tools")

@github_server.tool()
def list_repos(user: str) -> list:
    """List all repositories for a given user.

    Args:
        user: GitHub username to look up.

    Returns:
        List of repo dicts with name, language, stars, open_issues.
        Returns an error dict if the user is not found.
    """
    repos = MOCK_REPOS.get(user.lower())
    if repos is None:
        return [{"error": f"User '{user}' not found."}]
    return repos

@github_server.tool()
def get_file_content(repo: str, path: str) -> str:
    """Return the raw content of a file inside a repository.

    Args:
        repo: Repository name (e.g. 'web-app').
        path: File path within the repo (e.g. 'README.md').

    Returns:
        File content as a string, or an error message.
    """
    files = MOCK_FILES.get(repo)
    if files is None:
        return f"Error: repository '{repo}' not found."
    content = files.get(path)
    if content is None:
        return f"Error: '{path}' not found in '{repo}'."
    return content

@github_server.tool()
def list_issues(repo: str, state: str = "open") -> list:
    """List issues in a repository, optionally filtered by state.

    Args:
        repo: Repository name.
        state: Filter by issue state — 'open', 'closed', or 'all' (default 'open').

    Returns:
        List of issue dicts with id, title, state, labels.
    """
    all_issues = MOCK_ISSUES.get(repo, [])
    if state == "all":
        return all_issues
    return [i for i in all_issues if i["state"] == state]

@github_server.tool()
def search_code(repo: str, query: str) -> list:
    """Search for lines containing *query* across all files in a repository.

    Args:
        repo: Repository name to search inside.
        query: Case-insensitive substring to match.

    Returns:
        List of dicts with file, line_number, line_text for each match.
    """
    files = MOCK_FILES.get(repo)
    if files is None:
        return [{"error": f"Repository '{repo}' not found."}]
    matches = []
    for filename, content in files.items():
        for i, line in enumerate(content.splitlines(), start=1):
            if query.lower() in line.lower():
                matches.append({"file": filename, "line_number": i, "line_text": line})
    return matches

# %%
# Smoke-test all four tools directly
print("=== list_repos('alice') ===")
print(json.dumps(list_repos("alice"), indent=2))

print("\n=== get_file_content('web-app', 'app.py') ===")
print(get_file_content("web-app", "app.py"))

print("\n=== list_issues('web-app') ===")
print(json.dumps(list_issues("web-app"), indent=2))

print("\n=== search_code('web-app', 'flask') ===")
print(json.dumps(search_code("web-app", "flask"), indent=2))

assert len(list_repos("alice")) == 3, "Expected 3 repos for alice"
assert "Flask" in get_file_content("web-app", "app.py"), "Expected Flask in app.py"
assert len(list_issues("web-app", state="open")) == 2, "Expected 2 open issues"
assert len(search_code("web-app", "flask")) >= 1, "Expected at least 1 Flask match"
print("\nAll assertions passed.")

# %%
# Write the GitHub MCP server to a temp script for subprocess launch
GITHUB_SERVER_SCRIPT = Path(tempfile.gettempdir()) / "github_server.py"
GITHUB_SERVER_SCRIPT.write_text(f"""
import json
from mcp.server.fastmcp import FastMCP

MOCK_REPOS = {json.dumps(MOCK_REPOS)}
MOCK_FILES = {json.dumps(MOCK_FILES)}
MOCK_ISSUES = {json.dumps(MOCK_ISSUES)}

server = FastMCP("github-tools")

@server.tool()
def list_repos(user: str) -> list:
    \"\"\"List repositories for a user.\"\"\"
    return MOCK_REPOS.get(user.lower(), [{{"error": f"User not found"}}])

@server.tool()
def get_file_content(repo: str, path: str) -> str:
    \"\"\"Return raw file content from a repository.\"\"\"
    files = MOCK_FILES.get(repo)
    if not files:
        return f"Error: repo '{{repo}}' not found."
    return files.get(path, f"Error: '{{path}}' not found.")

@server.tool()
def list_issues(repo: str, state: str = "open") -> list:
    \"\"\"List issues in a repository filtered by state.\"\"\"
    issues = MOCK_ISSUES.get(repo, [])
    if state == "all":
        return issues
    return [i for i in issues if i["state"] == state]

@server.tool()
def search_code(repo: str, query: str) -> list:
    \"\"\"Search for a substring across all files in a repository.\"\"\"
    files = MOCK_FILES.get(repo)
    if not files:
        return [{{"error": f"Repo not found"}}]
    matches = []
    for fname, content in files.items():
        for i, line in enumerate(content.splitlines(), 1):
            if query.lower() in line.lower():
                matches.append({{"file": fname, "line_number": i, "line_text": line}})
    return matches

server.run()
""", encoding="utf-8")

print(f"GitHub server script written to: {GITHUB_SERVER_SCRIPT}")

# %%
async def run_github_lab(server_script: Path, question: str) -> str:
    """Launch the GitHub MCP server, discover tools, and answer a question via Mistral.

    Args:
        server_script: Path to the GitHub MCP server script.
        question: Natural language question to send to Mistral.

    Returns:
        Mistral's final answer string.
    """
    params = StdioServerParameters(command="python", args=[str(server_script)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_response = await session.list_tools()

            # Convert to Mistral format
            mistral_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema,
                    },
                }
                for t in tools_response.tools
            ]

            messages = [{"role": "user", "content": question}]
            start = time.time()

            # Agentic loop: keep calling tools until the model stops
            for _ in range(5):   # max 5 rounds
                response = client.chat.complete(
                    model="mistral-large-latest",
                    messages=messages,
                    tools=mistral_tools,
                    tool_choice="auto",
                )
                msg = response.choices[0].message

                if not msg.tool_calls:
                    elapsed = time.time() - start
                    print(f"Answered in {elapsed:.2f}s after {len(messages)} messages.")
                    return msg.content

                # Execute each tool call via MCP and append results
                messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})
                for tc in msg.tool_calls:
                    fn = tc.function
                    args = json.loads(fn.arguments) if isinstance(fn.arguments, str) else fn.arguments
                    print(f"  -> calling {fn.name}({args})")
                    result = await session.call_tool(fn.name, args)
                    tool_text = result.content[0].text if result.content else "[]"
                    messages.append({"role": "tool", "content": tool_text, "tool_call_id": tc.id})

    return "Loop ended without a final answer."

# %%
# Run the full lab
LAB_QUESTION = (
    "I'm looking at alice's GitHub account. "
    "What open issues does the web-app repo have, "
    "and can you show me the app.py file content?"
)

print(f"Question: {LAB_QUESTION}\n")
start = time.time()
try:
    lab_answer = asyncio.run(run_github_lab(GITHUB_SERVER_SCRIPT, LAB_QUESTION))
    print(f"\nFinal answer:\n{lab_answer}")
except Exception as exc:
    print(f"Lab error: {exc}")

# %% [markdown]
# ## Key Takeaways
# - MCP decouples tools from models: any MCP-compatible LLM can call your server without
#   code changes — just point it at the server URI.
# - FastMCP makes server creation as simple as decorating a Python function with
#   `@server.tool()`, handling serialisation and schema generation automatically.
# - Resources (via `@server.resource(uri)`) expose read-only data endpoints; dynamic
#   URI templates let you parameterise them like REST routes.
# - Always enforce access controls inside tool implementations (allowed-dir checks,
#   read-only SQL enforcement) — the LLM cannot be trusted to self-limit.
# - The MCP client + Mistral tool-calling loop forms a complete agentic pipeline:
#   discover tools, let Mistral decide which to call, execute via MCP, feed results
#   back, and repeat until the model returns a final answer.
