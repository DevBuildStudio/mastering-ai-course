# Workshop Training Material
## Master the FDE skills companies now expect

## Instructor Objectives
- Build confidence in production AI engineering decisions.
- Teach participants to diagnose failures, not just build happy-path demos.
- Connect technical design choices to customer and business outcomes.

## Chapter-Wise Delivery Plan
### Chapter 1: The FDE Mindset in AI Delivery (30 min)
- Outcome: Participants can explain why production AI delivery requires FDE-level ownership.
- Activity: Compare a demo architecture vs a production architecture.

### Chapter 2: LLM Behavior and System Constraints (45 min)
- Outcome: Participants can identify token and context constraints that shape architecture.
- Activity: Prompt budgeting exercise using fixed context limits.

### Chapter 3: Retrieval and RAG Failure Modes (60 min)
- Outcome: Participants can diagnose poor retrieval quality and propose fixes.
- Activity: Debug a broken retrieval pipeline with noisy chunks.

### Chapter 4: Tool Calling and Agent Control Loops (45 min)
- Outcome: Participants can design schema-safe tool calling flows.
- Activity: Draw planner-tool-validator loop and map failure points.

### Chapter 5: End-to-End Agent Implementation Lab (90 min)
- Outcome: Participants complete a working agent with retrieval and one tool integration.
- Activity: Build from `code/fde_agent_workshop.py` and validate outputs.

### Chapter 6: Guardrails, Safety, and HITL (45 min)
- Outcome: Participants implement at least one policy guardrail and one human approval gate.
- Activity: Red-team prompts and escalation policy exercise.

### Chapter 7: Evaluation and Observability (60 min)
- Outcome: Participants define an evaluation set and metrics dashboard fields.
- Activity: Analyze sample runs by latency, quality, and failure class.

### Chapter 8: Deployment on GCP and Vertex AI (60 min)
- Outcome: Participants create a deployment architecture with explicit trade-offs.
- Activity: Team architecture review using scenario cards.

### Chapter 9: Incident Response Playbook (30 min)
- Outcome: Participants can run first-response triage for production failures.
- Activity: Simulated outage (tool timeout + retrieval drift).

### Chapter 10: Capstone Build and Final Review (90 min)
- Outcome: Teams present a production-minded agent and defense of decisions.
- Activity: Demo, risk review, and peer evaluation.

## Teaching Notes by Module
### Module 1: AI Building Blocks
#### Concepts to Emphasize
- Token limits are product constraints, not just model details.
- Retrieval quality governs answer quality in enterprise settings.
- Structured outputs reduce downstream workflow failures.
- Guardrails are system-level, not prompt-only.

#### Common Misconceptions
- "Better model always solves retrieval issues." (False)
- "If it works once, it is production ready." (False)
- "Tool calling is deterministic." (False)

#### Whiteboard Activity
- Draw an agent loop with: user input -> planner -> tool call -> validator -> answer.
- Ask teams to identify where failures can happen and how to instrument each step.

### Module 2: End-to-End Agent Development
#### Demo Structure
1. Baseline agent with no retrieval.
2. Add retrieval over internal docs.
3. Add tool calling for enterprise APIs.
4. Add validation and fallback behavior.
5. Add simple evaluations and logs.

#### Facilitation Prompts
- "Where can this hallucinate?"
- "What if the API times out?"
- "How do we detect low-confidence answers?"
- "What should require human approval?"

#### Lab Success Criteria
- Agent returns an answer with source references.
- Tool call failures are handled without crashing.
- Logs include latency, tokens, and selected tool info.
- At least one HITL checkpoint is implemented.

### Module 3: Deploying Agents on GCP
#### Core Content
- Vertex AI endpoint patterns.
- Model Garden selection considerations.
- API gateway and authentication flow.
- Caching and cost controls.
- Reliability controls: retries, circuit breakers, timeouts.
- Monitoring dashboards and release gates.

#### Group Exercise
Provide three deployment scenarios and ask teams to choose architecture trade-offs:
- Low latency customer support assistant.
- High-accuracy internal policy assistant.
- Cost-sensitive high-volume triage assistant.

## Assessment Rubric
- Architecture clarity and reasoning: 25%
- Failure-mode handling: 25%
- Eval and observability quality: 20%
- Deployment strategy and trade-offs: 20%
- Communication and documentation: 10%

## Debrief Checklist
- Can participants explain why the system fails in edge cases?
- Can they describe mitigation plans?
- Can they justify cloud design trade-offs?
- Can they translate technical risk into business impact?

## Optional Extensions
- Add red-team prompts and safety evaluation.
- Add load simulation and SLO discussion.
- Add multi-agent routing and escalation patterns.
