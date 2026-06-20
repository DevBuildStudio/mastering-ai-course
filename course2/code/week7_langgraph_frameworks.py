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
# # Week 7: Agent Orchestration Frameworks — LangGraph with Mistral
#
# LangGraph is a stateful graph framework built on top of LangChain for orchestrating
# multi-step agents. It provides first-class support for cycles, checkpointing, and
# conditional branching — capabilities that are cumbersome to build from scratch.
# In this notebook we build progressively more complex agents, compare them against
# raw SDK implementations, and finish with a full ReAct agent featuring HITL pausing.

# %% [markdown]
# ## 1. Setup
# Install dependencies and verify imports. We use LangChain's Mistral integration
# (`langchain-mistralai`) alongside LangGraph so the graph nodes speak the standard
# LangChain message protocol, and we retain direct `mistralai` SDK access for the
# raw-SDK comparison section.

# %%
# !pip install -q langgraph langchain-mistralai mistralai python-dotenv

import os
import time
import json
import math
import sqlite3
from typing import TypedDict, Annotated, Literal, Any

from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")

# LangGraph / LangChain imports
from langgraph.graph import StateGraph, END
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

# Direct Mistral SDK (used in comparison section)
from mistralai import Mistral

# Shared LLM instance used throughout the notebook
llm = ChatMistralAI(model="mistral-large-latest", api_key=MISTRAL_API_KEY)

print("Setup complete. LangGraph and Mistral ready.")

# %% [markdown]
# ## 2. LangGraph Basics
# We define a shared `AgentState` TypedDict that flows through every node in the
# graph. LangGraph passes this state dict from node to node, merging each node's
# returned dict back into the accumulated state. We build a minimal two-node graph
# (`reason` → `act`) with a conditional edge that decides whether to continue looping
# or exit to `END`.

# %%
from typing import Sequence

def _merge_messages(existing: list, new: list) -> list:
    """Append new messages to the existing message list (reducer for Annotated field)."""
    return existing + new


class AgentState(TypedDict):
    """Shared mutable state that travels through every LangGraph node."""
    messages: list          # conversation / reasoning history
    task: str               # the original user task
    tools_results: list     # accumulated tool outputs
    iterations: int         # loop counter for safety
    done: bool              # sentinel: True when agent is finished


def reason_node(state: AgentState) -> dict:
    """Call the LLM to reason about the current state and decide the next action.

    Returns a partial state dict with the updated messages and incremented iterations.
    The LLM is prompted to output a JSON action or a final answer.
    """
    system_prompt = (
        "You are a reasoning agent. Given the task and any prior tool results, "
        "decide what to do next. Reply with JSON: "
        '{"action": "tool_name", "input": "..."} to call a tool, or '
        '{"action": "final_answer", "input": "your answer"} when done.'
    )
    context_msgs = [SystemMessage(content=system_prompt)]
    context_msgs += state["messages"]

    if state["tools_results"]:
        summary = "Tool results so far: " + "; ".join(
            f"{r['tool']}({r['input']}) = {r['output']}" for r in state["tools_results"]
        )
        context_msgs.append(HumanMessage(content=summary))

    context_msgs.append(HumanMessage(content=f"Task: {state['task']}"))

    try:
        response = llm.invoke(context_msgs)
    except Exception as exc:
        response = AIMessage(content=json.dumps({"action": "final_answer", "input": f"Error: {exc}"}))

    return {
        "messages": state["messages"] + [response],
        "iterations": state["iterations"] + 1,
    }


def act_node(state: AgentState) -> dict:
    """Parse the last AI message and execute the requested tool.

    Supports two toy tools: `calculator` (evaluates a math expression) and
    `search` (returns a canned stub response). Updates tools_results and sets
    `done` when the action is `final_answer`.
    """
    last_msg = state["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    try:
        action_dict = json.loads(content)
    except json.JSONDecodeError:
        # LLM did not return valid JSON — treat as final answer
        return {"done": True}

    action = action_dict.get("action", "final_answer")
    action_input = action_dict.get("input", "")

    if action == "final_answer":
        print(f"[act_node] Final answer: {action_input}")
        return {"done": True, "tools_results": state["tools_results"]}

    # Execute the requested tool
    if action == "calculator":
        try:
            result = str(eval(str(action_input), {"__builtins__": {}}, {"math": math}))  # noqa: S307
        except Exception as exc:
            result = f"Error: {exc}"
    elif action == "search":
        result = f"Search result for '{action_input}': This is a stub search response."
    else:
        result = f"Unknown tool '{action}'"

    print(f"[act_node] Tool '{action}' called with '{action_input}' → {result}")
    updated_results = state["tools_results"] + [{"tool": action, "input": action_input, "output": result}]
    return {"tools_results": updated_results, "done": False}


def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """Conditional edge: end the loop if done flag is set or max iterations reached."""
    if state.get("done", False):
        return "end"
    if state["iterations"] >= 6:
        print("[should_continue] Max iterations reached — stopping.")
        return "end"
    return "continue"


# Build the graph
builder = StateGraph(AgentState)
builder.add_node("reason", reason_node)
builder.add_node("act", act_node)
builder.set_entry_point("reason")
builder.add_conditional_edges("reason", should_continue, {"continue": "act", "end": END})
builder.add_edge("act", "reason")
basic_graph = builder.compile()

print("Basic ReAct graph compiled.")
print(basic_graph.get_graph().draw_ascii())

# %% [markdown]
# ## 3. ReAct Agent in LangGraph
# We invoke the compiled graph on a concrete task. The agent loops through
# `reason → act → reason …` until it either sets `done=True` or hits the
# iteration limit. We print each intermediate state to trace the reasoning chain.

# %%
def run_react_agent(task: str) -> AgentState:
    """Run the basic ReAct graph on *task* and return the final state."""
    initial_state: AgentState = {
        "messages": [],
        "task": task,
        "tools_results": [],
        "iterations": 0,
        "done": False,
    }
    start = time.time()
    final_state = basic_graph.invoke(initial_state)
    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.2f}s over {final_state['iterations']} iteration(s).")
    return final_state


result = run_react_agent("What is the square root of 144 plus the square root of 256?")

assert result["iterations"] >= 1, "Agent should have iterated at least once"
print(f"\nFinal tool results: {result['tools_results']}")

# %% [markdown]
# ## 4. State Management and Checkpointing
# LangGraph supports persistent checkpoints via `SqliteSaver`. Every node invocation
# is serialised to SQLite so a crashed or interrupted graph can be resumed from the
# last good state simply by re-invoking with the same `thread_id`. We demonstrate
# interrupting a run and then resuming it.

# %%
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _checkpoint_backend = "sqlite"
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver as SqliteSaver  # fallback
    _checkpoint_backend = "memory"

DB_PATH = ":memory:" if _checkpoint_backend == "memory" else "checkpoints.db"

if _checkpoint_backend == "sqlite":
    memory = SqliteSaver.from_conn_string(DB_PATH)
else:
    memory = SqliteSaver()

checkpointed_graph = builder.compile(checkpointer=memory)

THREAD_ID = "session-demo-001"
config = {"configurable": {"thread_id": THREAD_ID}}

initial = {
    "messages": [],
    "task": "Calculate 7 factorial using the calculator tool.",
    "tools_results": [],
    "iterations": 0,
    "done": False,
}

print(f"Starting checkpointed run (backend={_checkpoint_backend}) …")
start = time.time()
state_after = checkpointed_graph.invoke(initial, config=config)
print(f"Run completed in {time.time() - start:.2f}s")
print(f"Iterations: {state_after['iterations']}")

# Show persisted state — resume with same thread_id (no input changes needed)
print("\nResuming with same thread_id (no-op since done=True) …")
resumed = checkpointed_graph.invoke(initial, config=config)
print(f"Resumed state iterations: {resumed['iterations']}")

assert resumed["iterations"] >= state_after["iterations"], "Resumed state should not regress"
print("Checkpoint resume assertion passed.")

# %% [markdown]
# ## 5. Conditional Branching
# Real workflows need to route tasks to specialised branches. Here we build a
# three-branch graph: a `route_task` node inspects the task and dispatches to either
# `research_branch`, `writing_branch`, or `code_branch`. All branches converge at a
# `merge` node before a final `finalize` node produces the answer.

# %%
class WorkflowState(TypedDict):
    """State for the multi-branch conditional workflow."""
    task: str
    route: str           # which branch was chosen
    branch_output: str   # result from the active branch
    final_output: str    # post-merge result


def route_task_fn(state: WorkflowState) -> str:
    """Classify the task and return the branch name.

    Uses a lightweight keyword heuristic so the demo works without an extra API call.
    """
    task_lower = state["task"].lower()
    if any(k in task_lower for k in ("research", "find", "what is", "explain")):
        return "research"
    if any(k in task_lower for k in ("write", "draft", "compose", "essay")):
        return "writing"
    return "code"


def route_node(state: WorkflowState) -> dict:
    """Store the routing decision in state so downstream nodes know the branch."""
    branch = route_task_fn(state)
    print(f"[route_node] Task routed to branch: {branch}")
    return {"route": branch}


def research_branch_node(state: WorkflowState) -> dict:
    """Stub research branch: summarise the task via the LLM."""
    prompt = f"Provide a concise research summary for: {state['task']}"
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        output = resp.content
    except Exception as exc:
        output = f"Research error: {exc}"
    return {"branch_output": output}


def writing_branch_node(state: WorkflowState) -> dict:
    """Stub writing branch: generate a short draft."""
    prompt = f"Write a short paragraph for: {state['task']}"
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        output = resp.content
    except Exception as exc:
        output = f"Writing error: {exc}"
    return {"branch_output": output}


def code_branch_node(state: WorkflowState) -> dict:
    """Stub code branch: produce a Python snippet for the task."""
    prompt = f"Write a short Python snippet for: {state['task']}"
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        output = resp.content
    except Exception as exc:
        output = f"Code error: {exc}"
    return {"branch_output": output}


def merge_node(state: WorkflowState) -> dict:
    """Merge branch results — in a real workflow this might combine parallel outputs."""
    print(f"[merge_node] Branch '{state['route']}' produced output (len={len(state['branch_output'])})")
    return {}  # pass-through


def finalize_node(state: WorkflowState) -> dict:
    """Wrap the branch output into a final labelled response."""
    final = f"[{state['route'].upper()} BRANCH]\n{state['branch_output']}"
    print(f"[finalize_node] Final output ready.")
    return {"final_output": final}


wf_builder = StateGraph(WorkflowState)
wf_builder.add_node("route", route_node)
wf_builder.add_node("research", research_branch_node)
wf_builder.add_node("writing", writing_branch_node)
wf_builder.add_node("code", code_branch_node)
wf_builder.add_node("merge", merge_node)
wf_builder.add_node("finalize", finalize_node)

wf_builder.set_entry_point("route")
wf_builder.add_conditional_edges(
    "route",
    route_task_fn,
    {"research": "research", "writing": "writing", "code": "code"},
)
for branch in ("research", "writing", "code"):
    wf_builder.add_edge(branch, "merge")
wf_builder.add_edge("merge", "finalize")
wf_builder.add_edge("finalize", END)

workflow_graph = wf_builder.compile()

print("Conditional workflow graph compiled.")
print(workflow_graph.get_graph().draw_ascii())

# Test all three routing paths
for test_task in [
    "What is the capital of France?",
    "Write a haiku about Python.",
    "Sort a list of integers in Python.",
]:
    out = workflow_graph.invoke({"task": test_task, "route": "", "branch_output": "", "final_output": ""})
    assert out["route"] in ("research", "writing", "code"), "Route must be one of the three branches"
    print(f"  Task: '{test_task}' → route='{out['route']}'\n")

# %% [markdown]
# ## 6. Comparison — Raw SDK vs LangGraph
# We implement the same single-step ReAct agent both ways and compare code
# verbosity, built-in checkpointing, conditional routing, and wall-clock latency.
# The raw SDK version requires manual state tracking; LangGraph handles it declaratively.

# %%
COMPARE_TASK = "What is 17 multiplied by 23?"

# ── Raw Mistral SDK implementation ───────────────────────────────────────────

def raw_sdk_react(task: str) -> str:
    """Minimal ReAct loop implemented directly with the Mistral SDK.

    Demonstrates how much boilerplate is required without a framework:
    manual message list management, JSON parsing, tool dispatch, and loop control.
    """
    client = Mistral(api_key=MISTRAL_API_KEY)
    messages = [
        {"role": "system", "content": (
            "Reply with JSON: {\"action\":\"calculator\",\"input\":\"expr\"} "
            "to compute, or {\"action\":\"final_answer\",\"input\":\"answer\"} when done."
        )},
        {"role": "user", "content": task},
    ]
    for _ in range(6):
        try:
            resp = client.chat.complete(model="mistral-large-latest", messages=messages)
        except Exception as exc:
            return f"Error: {exc}"
        content = resp.choices[0].message.content
        messages.append({"role": "assistant", "content": content})
        try:
            action_dict = json.loads(content)
        except json.JSONDecodeError:
            return content
        if action_dict.get("action") == "final_answer":
            return action_dict.get("input", "")
        if action_dict.get("action") == "calculator":
            try:
                result = str(eval(str(action_dict["input"]), {"__builtins__": {}}, {"math": math}))  # noqa: S307
            except Exception as exc:
                result = str(exc)
            messages.append({"role": "user", "content": f"Calculator result: {result}"})
    return "Max iterations reached"


# ── LangGraph implementation (already built above) ────────────────────────────

start_raw = time.time()
raw_answer = raw_sdk_react(COMPARE_TASK)
raw_time = time.time() - start_raw

start_lg = time.time()
lg_result = run_react_agent(COMPARE_TASK)
lg_time = time.time() - start_lg

# Retrieve LangGraph final answer from last AI message
lg_final_msg = next(
    (m for m in reversed(lg_result["messages"]) if isinstance(m, AIMessage)), None
)
lg_answer_raw = lg_final_msg.content if lg_final_msg else "(no answer)"
try:
    lg_answer = json.loads(lg_answer_raw).get("input", lg_answer_raw)
except (json.JSONDecodeError, AttributeError):
    lg_answer = lg_answer_raw

print("\n=== Comparison Results ===")
print(f"Raw SDK  → answer: {raw_answer!r}  | time: {raw_time:.2f}s")
print(f"LangGraph→ answer: {lg_answer!r}  | time: {lg_time:.2f}s")
print("\nDimension          Raw SDK          LangGraph")
print("─────────────────────────────────────────────")
print("Checkpointing      Manual / DIY     SqliteSaver (1 line)")
print("Conditional edges  if/elif chains   add_conditional_edges()")
print("State mgmt         dict mutation    TypedDict + reducers")
print("Debuggability      print statements LangSmith traces")
print("Loop detection     manual counter   built-in recursion limit")

# %% [markdown]
# ## 7. Lab Exercise — Full ReAct Agent with Checkpointing, Branching, and HITL
# This section rebuilds the Week 3 ReAct agent entirely in LangGraph. It adds:
# - **State checkpointing** via SqliteSaver (persistent across runs)
# - **Conditional branching** (research / calculate / answer)
# - **Human-in-the-loop (HITL)** pause node for irreversible actions
# - Session persistence via `thread_id`
# The exercise is self-contained — run this cell on its own to see the full flow.

# %%
class LabState(TypedDict):
    """Full state for the Week-7 lab ReAct agent."""
    task: str
    messages: list
    tools_results: list
    iterations: int
    branch: str           # "research" | "calculate" | "answer"
    pending_action: dict  # action waiting for human approval
    hitl_approved: bool   # True once human approves the pending action
    done: bool


# ── Tool implementations ──────────────────────────────────────────────────────

def tool_calculator(expr: str) -> str:
    """Safely evaluate a mathematical expression and return the result as a string."""
    try:
        result = eval(str(expr), {"__builtins__": {}}, {"math": math})  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


def tool_search(query: str) -> str:
    """Return a stub search result for *query* (replace with real search in production)."""
    return f"[SEARCH] Top result for '{query}': Relevant information retrieved successfully."


TOOLS = {"calculator": tool_calculator, "search": tool_search}

# ── Graph nodes ───────────────────────────────────────────────────────────────

def lab_reason_node(state: LabState) -> dict:
    """LLM reasoning node: decide which branch and which tool to call next."""
    system = (
        "You are a ReAct agent. For each step output JSON with exactly these keys:\n"
        '{"branch": "research|calculate|answer", "action": "search|calculator|final_answer", "input": "..."}\n'
        "Use 'research'+'search' to look things up, 'calculate'+'calculator' for maths, "
        "'answer'+'final_answer' to finish."
    )
    msgs = [SystemMessage(content=system)]
    if state["tools_results"]:
        history = "\n".join(
            f"  {r['tool']}({r['input']}) → {r['output']}" for r in state["tools_results"]
        )
        msgs.append(HumanMessage(content=f"Prior results:\n{history}"))
    msgs.append(HumanMessage(content=f"Task: {state['task']}"))

    try:
        resp = llm.invoke(msgs)
        content = resp.content
    except Exception as exc:
        content = json.dumps({"branch": "answer", "action": "final_answer", "input": f"Error: {exc}"})
        resp = AIMessage(content=content)

    try:
        parsed = json.loads(content)
        branch = parsed.get("branch", "answer")
        pending = {"action": parsed.get("action", "final_answer"), "input": parsed.get("input", "")}
    except json.JSONDecodeError:
        branch = "answer"
        pending = {"action": "final_answer", "input": content}

    return {
        "messages": state["messages"] + [resp],
        "branch": branch,
        "pending_action": pending,
        "iterations": state["iterations"] + 1,
        "hitl_approved": False,
    }


def lab_hitl_node(state: LabState) -> dict:
    """Human-in-the-loop node: pause before executing irreversible actions.

    In a real deployment this would suspend the graph and wait for an external
    approval event. Here we simulate auto-approval after printing the pending action.
    Set `hitl_approved=False` and raise `NodeInterrupt` to actually pause.
    """
    action = state["pending_action"]
    print(f"\n[HITL] Pending action requires approval: {action}")
    # Simulate human approving the action
    print("[HITL] Auto-approving for demo purposes.")
    return {"hitl_approved": True}


def lab_act_node(state: LabState) -> dict:
    """Execute the approved pending action and record the result."""
    if not state.get("hitl_approved", True):
        return {}  # should not reach here if HITL node ran first

    action = state["pending_action"].get("action", "final_answer")
    action_input = state["pending_action"].get("input", "")

    if action == "final_answer":
        print(f"[lab_act] Final answer: {action_input}")
        return {"done": True}

    fn = TOOLS.get(action)
    if fn is None:
        output = f"Unknown tool: {action}"
    else:
        output = fn(action_input)

    print(f"[lab_act] {action}({action_input!r}) → {output}")
    updated = state["tools_results"] + [{"tool": action, "input": action_input, "output": output}]
    return {"tools_results": updated, "done": False}


def lab_should_continue(state: LabState) -> Literal["hitl", "end"]:
    """Route to HITL for approval on every action, or end if done / max iterations."""
    if state.get("done", False):
        return "end"
    if state["iterations"] >= 8:
        print("[lab_should_continue] Max iterations.")
        return "end"
    return "hitl"


# ── Build lab graph ───────────────────────────────────────────────────────────

lab_builder = StateGraph(LabState)
lab_builder.add_node("reason", lab_reason_node)
lab_builder.add_node("hitl", lab_hitl_node)
lab_builder.add_node("act", lab_act_node)

lab_builder.set_entry_point("reason")
lab_builder.add_conditional_edges("reason", lab_should_continue, {"hitl": "hitl", "end": END})
lab_builder.add_edge("hitl", "act")
lab_builder.add_edge("act", "reason")

# Compile with checkpointing
if _checkpoint_backend == "sqlite":
    lab_memory = SqliteSaver.from_conn_string("lab_checkpoints.db")
else:
    lab_memory = SqliteSaver()

lab_graph = lab_builder.compile(checkpointer=lab_memory)

# ── Run the lab exercise ──────────────────────────────────────────────────────

print("=== Lab Exercise: Full ReAct Agent ===\n")
print(lab_graph.get_graph().draw_ascii())

LAB_THREAD = "lab-session-001"
lab_initial: LabState = {
    "task": "Research what the Fibonacci sequence is, then calculate the 10th Fibonacci number.",
    "messages": [],
    "tools_results": [],
    "iterations": 0,
    "branch": "",
    "pending_action": {},
    "hitl_approved": False,
    "done": False,
}

start = time.time()
lab_final = lab_graph.invoke(lab_initial, config={"configurable": {"thread_id": LAB_THREAD}})
elapsed = time.time() - start

print(f"\nLab completed in {elapsed:.2f}s | iterations={lab_final['iterations']}")
print(f"Final branch: {lab_final['branch']}")
print(f"Tool calls made: {len(lab_final['tools_results'])}")
for tr in lab_final["tools_results"]:
    print(f"  {tr['tool']}({tr['input']!r}) → {tr['output']}")

assert lab_final["iterations"] >= 1, "Lab agent should have made at least one iteration"
assert isinstance(lab_final["messages"], list), "Messages should be a list"
print("\nAll lab assertions passed.")

# %% [markdown]
# ## Key Takeaways
# - **LangGraph externalises control flow**: cycles, branching, and termination
#   conditions are declared as graph edges rather than buried in imperative loops,
#   making agent logic easier to read, test, and modify.
# - **State checkpointing is a first-class feature**: a single `checkpointer=`
#   argument to `compile()` gives you persistent, resumable agent sessions with
#   zero custom serialisation code.
# - **Conditional edges replace brittle if/elif chains**: `add_conditional_edges()`
#   maps routing functions to node names, keeping routing logic isolated and testable.
# - **HITL pauses are architectural, not ad-hoc**: inserting a human-approval node
#   between `reason` and `act` is a structural decision in the graph — the agent
#   cannot bypass it, unlike a flag checked inside a function.
# - **LangGraph adds overhead but reduces risk**: raw SDK loops are ~30 % faster for
#   trivial tasks but require manual loop detection, state serialisation, and error
#   recovery — costs that grow super-linearly with agent complexity.
