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
# # Course 2, Week 4: Planning and Complex Reasoning
#
# This notebook explores how LLMs can plan, reason over multi-step problems,
# and dynamically adapt when things go wrong. We cover task decomposition,
# Tree of Thought (ToT), dynamic replanning, and self-critique loops using
# the Mistral API.

# %% [markdown]
# ## 1. Setup
# Core imports, Mistral client initialization, and mock tool definitions
# for web search, file writing, and calculator operations.

# %%
import os
import json
import time
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

from mistralai import Mistral
from mistralai.models import SDKError

client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY", "your-key-here"))

# --- Mock tools ---

def web_search(query: str) -> str:
    """Simulate a web search returning mock results."""
    mock_results = {
        "python 3.13": (
            "Python 3.13 introduces a new interactive interpreter (REPL) with "
            "multi-line editing, experimental free-threaded mode (no GIL via "
            "--disable-gil), a new JIT compiler option (--enable-experimental-jit), "
            "improved error messages with color support, and deprecation of several "
            "legacy stdlib modules. Released October 2024."
        ),
        "default": f"Search results for '{query}': [mock data] relevant information found."
    }
    for key, val in mock_results.items():
        if key in query.lower():
            return val
    return mock_results["default"]

def file_write(filename: str, content: str) -> str:
    """Write content to a file and return a status message."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Success: wrote {len(content)} characters to '{filename}'."
    except OSError as e:
        return f"Error writing file: {e}"

def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression string."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters."
    try:
        result = eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

TOOLS_MAP = {
    "web_search": web_search,
    "file_write": file_write,
    "calculator": calculator,
}

print("Setup complete. Tools registered:", list(TOOLS_MAP.keys()))

# %% [markdown]
# ## 2. Task Planner
# The `TaskPlanner` uses Mistral to decompose a high-level goal into ordered
# steps with explicit tool assignments and dependency tracking. Plans are
# represented as dataclasses and serialized to JSON for inspection.

# %%
PLANNER_PROMPT = """\
You are a task planner. Given a goal, decompose it into concrete steps.
Each step must specify which tool to use (web_search, file_write, calculator, or none).
Return ONLY valid JSON matching this schema exactly:
{{
  "goal": "<original goal>",
  "steps": [
    {{
      "id": 1,
      "description": "<what this step does>",
      "depends_on": [],
      "tool": "<tool name or null>",
      "tool_input": "<input string for the tool>"
    }}
  ]
}}
Available tools: web_search(query), file_write(filename, content), calculator(expression).
For file_write steps, tool_input must be "filename|||content".
Keep the plan to 3-5 steps. Goal: {goal}
"""

@dataclass
class Step:
    """A single executable step in a plan."""
    id: int
    description: str
    depends_on: List[int]
    tool: Optional[str]
    tool_input: str = ""
    status: str = "pending"   # pending | running | done | failed
    result: Optional[str] = None

@dataclass
class Plan:
    """A complete execution plan for a goal."""
    goal: str
    steps: List[Step]
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

class TaskPlanner:
    """Decompose a user goal into an ordered, tool-annotated plan using Mistral."""

    def __init__(self, model: str = "mistral-large-latest"):
        """Initialize with the chosen Mistral model."""
        self.model = model

    def create_plan(self, task: str) -> Plan:
        """Call Mistral to generate a Plan for the given task string."""
        prompt = PLANNER_PROMPT.format(goal=task)
        try:
            resp = client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw)
        except (SDKError, json.JSONDecodeError) as e:
            print(f"Planner error: {e}. Using fallback plan.")
            data = {
                "goal": task,
                "steps": [{"id": 1, "description": task, "depends_on": [],
                            "tool": None, "tool_input": ""}]
            }
        steps = [
            Step(
                id=s["id"],
                description=s["description"],
                depends_on=s.get("depends_on", []),
                tool=s.get("tool"),
                tool_input=s.get("tool_input", ""),
            )
            for s in data["steps"]
        ]
        return Plan(goal=data["goal"], steps=steps)

    def validate_plan(self, plan: Plan) -> bool:
        """Check the plan for dependency cycles using DFS. Returns True if valid."""
        ids = {s.id for s in plan.steps}
        adj: Dict[int, List[int]] = {s.id: s.depends_on for s in plan.steps}
        visited, in_stack = set(), set()

        def dfs(node: int) -> bool:
            if node in in_stack:
                return True   # cycle detected
            if node in visited:
                return False
            visited.add(node)
            in_stack.add(node)
            for nb in adj.get(node, []):
                if nb in ids and dfs(nb):
                    return True
            in_stack.discard(node)
            return False

        has_cycle = any(dfs(s.id) for s in plan.steps if s.id not in visited)
        if has_cycle:
            print("WARNING: cycle detected in plan!")
        return not has_cycle

planner = TaskPlanner()
start = time.time()
sample_plan = planner.create_plan("Calculate the area of a circle with radius 7 and save it to area.txt")
elapsed = time.time() - start

print(f"Plan created in {elapsed:.2f}s")
print(f"Goal: {sample_plan.goal}")
for s in sample_plan.steps:
    print(f"  Step {s.id}: [{s.tool or 'none'}] {s.description}")

assert planner.validate_plan(sample_plan), "Plan has cycles!"
print("Plan is valid (no cycles).")

# %% [markdown]
# ## 3. Plan Executor
# `PlanExecutor` iterates steps in topological order, calling the appropriate
# tool for each step. It retries failed steps once, then marks them as failed
# and skips any dependent steps, recording the full `ExecutionTrace`.

# %%
@dataclass
class ExecutionTrace:
    """Record of all step executions including timing and results."""
    plan_goal: str
    events: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, step_id: int, status: str, result: str, duration: float) -> None:
        """Append a step execution event to the trace."""
        self.events.append({
            "step_id": step_id, "status": status,
            "result": result[:200], "duration_s": round(duration, 3)
        })

class PlanExecutor:
    """Execute a Plan step-by-step, handling retries and dependency skipping."""

    def run_step(self, step: Step) -> str:
        """Dispatch a step to its registered tool and return the result string."""
        if not step.tool or step.tool not in TOOLS_MAP:
            # No tool — ask Mistral to fulfill the step directly
            try:
                resp = client.chat.complete(
                    model="mistral-small-latest",
                    messages=[{"role": "user", "content": step.description}],
                )
                return resp.choices[0].message.content.strip()
            except SDKError as e:
                return f"LLM step error: {e}"

        tool_fn = TOOLS_MAP[step.tool]
        inp = step.tool_input

        if step.tool == "file_write":
            parts = inp.split("|||", 1)
            if len(parts) == 2:
                return tool_fn(parts[0].strip(), parts[1].strip())
            return "Error: file_write needs 'filename|||content' format."
        else:
            return tool_fn(inp)

    def topological_order(self, plan: Plan) -> List[Step]:
        """Return steps sorted so dependencies come before dependents."""
        step_map = {s.id: s for s in plan.steps}
        visited, order = set(), []

        def visit(sid: int) -> None:
            if sid in visited:
                return
            for dep in step_map[sid].depends_on:
                if dep in step_map:
                    visit(dep)
            visited.add(sid)
            order.append(step_map[sid])

        for s in plan.steps:
            visit(s.id)
        return order

    def execute_plan(self, plan: Plan) -> ExecutionTrace:
        """Execute all steps in dependency order, retrying failures once."""
        trace = ExecutionTrace(plan_goal=plan.goal)
        failed_ids = set()
        ordered = self.topological_order(plan)

        for step in ordered:
            # Skip if any dependency failed
            if any(d in failed_ids for d in step.depends_on):
                step.status = "failed"
                step.result = "Skipped: dependency failed."
                trace.record(step.id, "skipped", step.result, 0.0)
                print(f"  Step {step.id} SKIPPED (dependency failed)")
                failed_ids.add(step.id)
                continue

            step.status = "running"
            print(f"  Step {step.id}: {step.description[:60]}...")

            for attempt in range(2):
                t0 = time.time()
                try:
                    result = self.run_step(step)
                    step.status = "done"
                    step.result = result
                    trace.record(step.id, "done", result, time.time() - t0)
                    print(f"    -> done ({time.time()-t0:.2f}s): {result[:80]}")
                    break
                except Exception as e:
                    if attempt == 0:
                        print(f"    -> attempt 1 failed ({e}), retrying...")
                        time.sleep(0.5)
                    else:
                        step.status = "failed"
                        step.result = str(e)
                        trace.record(step.id, "failed", str(e), time.time() - t0)
                        print(f"    -> FAILED: {e}")
                        failed_ids.add(step.id)

        print(f"\nExecution complete. Steps done/failed: "
              f"{sum(1 for s in plan.steps if s.status=='done')}/"
              f"{sum(1 for s in plan.steps if s.status=='failed')}")
        return trace

executor = PlanExecutor()
print(f"\nExecuting plan: {sample_plan.goal}")
trace = executor.execute_plan(sample_plan)

# %% [markdown]
# ## 4. Tree of Thought
# `TreeOfThought` generates multiple reasoning branches for a problem, scores
# each branch against evaluation criteria (feasibility, completeness, risk),
# selects the best branch, and supports backtracking when a branch is invalid.

# %%
@dataclass
class Branch:
    """One reasoning branch in a Tree of Thought exploration."""
    id: int
    reasoning: str
    proposed_solution: str
    scores: Dict[str, float] = field(default_factory=dict)

    def total_score(self) -> float:
        """Return the sum of all evaluation scores."""
        return sum(self.scores.values())

class TreeOfThought:
    """Explore a problem via multiple reasoning branches and score each one."""

    def __init__(self, model: str = "mistral-large-latest"):
        """Initialize with the Mistral model to use."""
        self.model = model

    def generate_branches(self, problem: str, n_branches: int = 3) -> List[Branch]:
        """Ask Mistral to generate n distinct reasoning approaches for the problem."""
        prompt = (
            f"Problem: {problem}\n\n"
            f"Generate {n_branches} DISTINCT approaches to solve this problem. "
            "For each approach, provide step-by-step reasoning and a proposed solution. "
            "Return ONLY valid JSON: "
            '{"branches": [{"id": 1, "reasoning": "...", "proposed_solution": "..."}, ...]}'
        )
        try:
            resp = client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return [Branch(id=b["id"], reasoning=b["reasoning"],
                           proposed_solution=b["proposed_solution"])
                    for b in data["branches"]]
        except (SDKError, json.JSONDecodeError) as e:
            print(f"Branch generation error: {e}")
            return [Branch(id=1, reasoning="Direct approach.", proposed_solution=problem)]

    def score_branch(self, branch: Branch, problem: str) -> Branch:
        """Score a branch on feasibility, completeness, and risk (each 1-5)."""
        prompt = (
            f"Problem: {problem}\n"
            f"Proposed solution reasoning: {branch.reasoning}\n"
            f"Proposed solution: {branch.proposed_solution}\n\n"
            "Score this solution on three criteria (integers 1-5 each):\n"
            "- feasibility: how realistic is this solution?\n"
            "- completeness: does it fully address the problem?\n"
            "- risk: how low is the risk of failure? (5=very low risk)\n"
            'Return ONLY valid JSON: {"feasibility": N, "completeness": N, "risk": N}'
        )
        try:
            resp = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            scores = json.loads(resp.choices[0].message.content)
            branch.scores = {k: float(v) for k, v in scores.items()}
        except (SDKError, json.JSONDecodeError) as e:
            print(f"Scoring error for branch {branch.id}: {e}")
            branch.scores = {"feasibility": 3.0, "completeness": 3.0, "risk": 3.0}
        return branch

    def select_best_branch(self, branches: List[Branch]) -> Branch:
        """Return the branch with the highest total score."""
        return max(branches, key=lambda b: b.total_score())

    def backtrack(self, branch: Branch) -> str:
        """Return a brief explanation of why this branch should be abandoned."""
        return (f"Branch {branch.id} backtracked. "
                f"Total score {branch.total_score():.1f}/15 was insufficient. "
                "Trying alternative branch.")

# Demo: logic puzzle
puzzle = (
    "A farmer needs to cross a river with a fox, a chicken, and a bag of grain. "
    "The boat holds only the farmer and one item. The fox eats the chicken if left alone; "
    "the chicken eats the grain if left alone. How does the farmer get all across safely?"
)

print("Tree of Thought — Logic Puzzle")
print(f"Problem: {puzzle[:80]}...\n")

tot = TreeOfThought()
start = time.time()
branches = tot.generate_branches(puzzle, n_branches=3)
print(f"Generated {len(branches)} branches in {time.time()-start:.2f}s")

for b in branches:
    tot.score_branch(b, puzzle)
    print(f"  Branch {b.id}: scores={b.scores} total={b.total_score():.1f}")

best = tot.select_best_branch(branches)
print(f"\nBest branch: #{best.id} (score {best.total_score():.1f}/15)")
print(f"Solution: {best.proposed_solution[:200]}")

# Demonstrate backtracking on the lowest-scoring branch
worst = min(branches, key=lambda b: b.total_score())
print(f"\n{tot.backtrack(worst)}")

# %% [markdown]
# ## 5. Dynamic Replanning
# `DynamicPlanner` extends `TaskPlanner` to handle step failures gracefully.
# When a step fails it calls `replan_from` to produce a revised plan, detects
# whether the overall goal is still achievable, and calls a partial success
# handler when at least some steps have already succeeded.

# %%
class DynamicPlanner(TaskPlanner):
    """TaskPlanner with failure-driven replanning capabilities."""

    def replan_from(self, failed_step: Step, context: Dict[str, Any]) -> Plan:
        """Generate a revised plan starting from a failed step, given prior context."""
        completed = context.get("completed_steps", [])
        completed_text = "; ".join(completed) if completed else "none"
        prompt = (
            f"Original goal: {context.get('goal', 'unknown')}\n"
            f"Completed steps so far: {completed_text}\n"
            f"Failed step: '{failed_step.description}'\n"
            f"Failure reason: {failed_step.result}\n\n"
            "Create a REVISED plan to still achieve the original goal, "
            "working around the failure. Return JSON matching the same schema as before.\n"
            "Schema: {\"goal\": \"...\", \"steps\": [{\"id\": 1, \"description\": \"...\", "
            "\"depends_on\": [], \"tool\": \"...\", \"tool_input\": \"...\"}]}"
        )
        try:
            resp = client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            steps = [
                Step(id=s["id"], description=s["description"],
                     depends_on=s.get("depends_on", []),
                     tool=s.get("tool"), tool_input=s.get("tool_input", ""))
                for s in data["steps"]
            ]
            return Plan(goal=data["goal"], steps=steps)
        except (SDKError, json.JSONDecodeError) as e:
            print(f"Replan error: {e}")
            return Plan(goal=context.get("goal", "unknown"), steps=[])

    def is_goal_achievable(self, plan: Plan, failed_ids: set) -> bool:
        """Return False if all remaining steps depend on a failed step."""
        remaining = [s for s in plan.steps if s.status not in ("done",)]
        if not remaining:
            return True
        achievable = any(
            not any(d in failed_ids for d in s.depends_on)
            for s in remaining
        )
        return achievable

    def partial_success_handler(self, plan: Plan) -> str:
        """Summarize which steps succeeded and which failed for the user."""
        done = [s for s in plan.steps if s.status == "done"]
        failed = [s for s in plan.steps if s.status == "failed"]
        msg = (f"Partial success: {len(done)}/{len(plan.steps)} steps completed. "
               f"Completed: {[s.id for s in done]}. Failed: {[s.id for s in failed]}.")
        return msg

# Demo: replanning when web search returns 0 results
print("Dynamic Replanning Demo\n")
dyn_planner = DynamicPlanner()
goal = "Find today's top AI news, summarize it, and save to ai_news.txt"
plan_dyn = dyn_planner.create_plan(goal)
print(f"Original plan ({len(plan_dyn.steps)} steps):")
for s in plan_dyn.steps:
    print(f"  Step {s.id}: {s.description}")

# Simulate step 1 (web search) returning empty results
if plan_dyn.steps:
    failed_step = plan_dyn.steps[0]
    failed_step.status = "failed"
    failed_step.result = "web_search returned 0 results for query."

    context = {
        "goal": goal,
        "completed_steps": [],
    }
    start = time.time()
    revised = dyn_planner.replan_from(failed_step, context)
    print(f"\nRevised plan generated in {time.time()-start:.2f}s ({len(revised.steps)} steps):")
    for s in revised.steps:
        print(f"  Step {s.id}: {s.description}")

    print("\n" + dyn_planner.partial_success_handler(plan_dyn))

# %% [markdown]
# ## 6. Self-Critique and Reflexion
# The `Reflexion` class implements a critique-then-improve loop: it asks
# Mistral to score an output, surface issues, and suggest improvements, then
# rewrites the output — iterating up to `max_rounds` times and tracking the
# score trajectory.

# %%
CRITIQUE_PROMPT = """\
You are a strict quality evaluator. Given a task and an output, evaluate the output.
Return ONLY valid JSON:
{{
  "score": <integer 1-10>,
  "issues": ["issue 1", "issue 2"],
  "improvements": ["improvement 1", "improvement 2"]
}}
Task: {task}
Output to evaluate:
{output}
"""

class Reflexion:
    """Self-critique and iterative improvement loop using Mistral."""

    def __init__(self, model: str = "mistral-large-latest"):
        """Initialize with the Mistral model to use for critique and improvement."""
        self.model = model

    def critique_output(self, task: str, output: str) -> Dict[str, Any]:
        """Ask Mistral to critique the output and return score, issues, improvements."""
        prompt = CRITIQUE_PROMPT.format(task=task, output=output[:1500])
        try:
            resp = client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except (SDKError, json.JSONDecodeError) as e:
            print(f"Critique error: {e}")
            return {"score": 5, "issues": [], "improvements": []}

    def improve_output(self, task: str, output: str, critique: Dict[str, Any]) -> str:
        """Rewrite output addressing the issues raised in the critique."""
        issues_text = "\n".join(f"- {i}" for i in critique.get("issues", []))
        impr_text = "\n".join(f"- {i}" for i in critique.get("improvements", []))
        prompt = (
            f"Task: {task}\n\n"
            f"Current output:\n{output}\n\n"
            f"Issues found:\n{issues_text}\n\n"
            f"Required improvements:\n{impr_text}\n\n"
            "Please rewrite the output addressing all issues and improvements. "
            "Return only the improved output text, no preamble."
        )
        try:
            resp = client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        except SDKError as e:
            print(f"Improve error: {e}")
            return output

    def reflexion_loop(self, task: str, max_rounds: int = 3) -> Dict[str, Any]:
        """Run the full reflexion loop: generate -> critique -> improve, up to max_rounds."""
        # Initial generation
        try:
            resp = client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": task}],
            )
            current_output = resp.choices[0].message.content.strip()
        except SDKError as e:
            return {"error": str(e), "trajectory": []}

        trajectory = []
        print(f"Reflexion loop for: {task[:60]}...")

        for round_num in range(1, max_rounds + 1):
            critique = self.critique_output(task, current_output)
            score = critique.get("score", 5)
            trajectory.append({"round": round_num, "score": score,
                                "issues_count": len(critique.get("issues", []))})
            print(f"  Round {round_num}: score={score}/10, "
                  f"issues={len(critique.get('issues', []))}")

            if score >= 9:
                print(f"  -> Score {score}/10 reached threshold. Stopping early.")
                break

            if round_num < max_rounds:
                current_output = self.improve_output(task, current_output, critique)

        print(f"Score trajectory: {[t['score'] for t in trajectory]}")
        return {"final_output": current_output, "trajectory": trajectory,
                "final_score": trajectory[-1]["score"]}

reflexion = Reflexion()
start = time.time()
result = reflexion.reflexion_loop(
    "Explain why Python's GIL exists and how Python 3.13 addresses it.",
    max_rounds=3
)
print(f"\nCompleted in {time.time()-start:.2f}s")
print(f"Final score: {result['final_score']}/10")
assert result["final_score"] >= 1, "Score should be between 1 and 10"
print(f"Final output preview: {result['final_output'][:200]}...")

# %% [markdown]
# ## 7. Lab Exercise
# Build a complete plan-and-execute agent that: researches Python 3.13 features,
# writes a 300-word summary, saves it to `python313_summary.txt`, then evaluates
# the file contents — with a self-critique pass applied to the final summary.

# %%
def lab_plan_and_execute() -> None:
    """
    Lab: Full plan-and-execute pipeline with self-critique.

    Steps:
    1. Plan the research/write/evaluate task using TaskPlanner.
    2. Execute the plan step by step using PlanExecutor.
    3. Apply a Reflexion critique pass to the generated summary.
    4. Print the plan, execution trace, and critique scores.
    """
    print("=" * 60)
    print("LAB: Plan-and-Execute Agent — Python 3.13 Research")
    print("=" * 60)

    # --- Step A: Create the plan ---
    lab_planner = TaskPlanner()
    lab_goal = (
        "Research the latest Python 3.13 features using web_search, "
        "write a 300-word summary of those features, save the summary to "
        "python313_summary.txt using file_write, and then verify the file "
        "was written successfully."
    )

    print("\n[1] Creating plan...")
    t0 = time.time()
    plan = lab_planner.create_plan(lab_goal)
    print(f"    Plan created in {time.time()-t0:.2f}s ({len(plan.steps)} steps)")
    valid = lab_planner.validate_plan(plan)
    print(f"    Plan valid (no cycles): {valid}")

    print("\n    Plan steps:")
    for s in plan.steps:
        deps = f" (depends on {s.depends_on})" if s.depends_on else ""
        print(f"      Step {s.id} [{s.tool or 'llm'}]{deps}: {s.description}")

    # --- Step B: Execute the plan ---
    print("\n[2] Executing plan...")
    lab_executor = PlanExecutor()
    trace = lab_executor.execute_plan(plan)

    print("\n    Execution trace:")
    for event in trace.events:
        print(f"      Step {event['step_id']}: {event['status'].upper()} "
              f"({event['duration_s']}s) -> {event['result'][:80]}")

    # --- Step C: Extract generated summary for critique ---
    summary_text = ""
    for step in plan.steps:
        if step.result and "python" in step.description.lower():
            summary_text = step.result
            break
    # Fallback: use web_search mock content
    if not summary_text:
        summary_text = web_search("python 3.13")

    # --- Step D: Self-critique pass ---
    print("\n[3] Running self-critique on the summary...")
    reflexion_lab = Reflexion()
    t0 = time.time()
    critique = reflexion_lab.critique_output(
        task="Write a clear, accurate 300-word summary of Python 3.13 new features.",
        output=summary_text,
    )
    print(f"    Critique completed in {time.time()-t0:.2f}s")
    print(f"    Score:        {critique.get('score', 'N/A')}/10")
    print(f"    Issues:       {critique.get('issues', [])}")
    print(f"    Improvements: {critique.get('improvements', [])}")

    if critique.get("score", 0) < 8:
        print("\n    Applying one improvement round...")
        improved = reflexion_lab.improve_output(
            "Write a clear, accurate 300-word summary of Python 3.13 new features.",
            summary_text, critique
        )
        # Save improved version
        save_result = file_write("python313_summary.txt", improved)
        print(f"    Saved improved summary: {save_result}")
        final_critique = reflexion_lab.critique_output(
            "Write a clear, accurate 300-word summary of Python 3.13 new features.",
            improved,
        )
        print(f"    Improved score: {final_critique.get('score', 'N/A')}/10")
    else:
        print("    Summary quality is already good (score >= 8).")

    # --- Step E: Final summary ---
    done_count = sum(1 for s in plan.steps if s.status == "done")
    print(f"\n[4] Summary: {done_count}/{len(plan.steps)} steps succeeded.")
    print("    Lab exercise complete.")
    print("=" * 60)

# Run the lab
lab_plan_and_execute()

# %% [markdown]
# ## Key Takeaways
# - **Structured planning** transforms vague goals into verifiable, dependency-aware
#   step graphs — enabling reliable automation and clear failure attribution.
# - **Tree of Thought** explores multiple reasoning paths in parallel and scores
#   each, dramatically improving solution quality for complex logic problems.
# - **Dynamic replanning** makes agents resilient: instead of stopping at the first
#   failure, the system repairs the plan and continues toward the original goal.
# - **Reflexion (self-critique)** creates a quality feedback loop — the model
#   evaluates its own output and iteratively rewrites it, raising scores without
#   human intervention.
# - **ExecutionTrace and validation** (cycle detection, dependency skipping, retry
#   logic) are essential engineering primitives for any production planning agent.
