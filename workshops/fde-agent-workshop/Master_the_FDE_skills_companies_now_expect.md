# Master the FDE skills companies now expect

## Workshop Overview
Forward Deployed Engineering is becoming one of the most important roles in AI because companies are no longer asking, "Can we build a demo?" They are asking, "Can this agent work inside our real customer environment?"

That shift demands a broader engineering skill set:
- Understanding LLM behavior and limitations.
- Diagnosing and hardening RAG systems.
- Designing robust tool-calling agent workflows.
- Integrating enterprise APIs and internal knowledge bases.
- Deploying to production with clear constraints on latency, cost, scale, security, and reliability.

This workshop bridges that gap end-to-end. You will go from AI fundamentals to building a working agent, connecting it to real tools and enterprise systems, and deploying it to Google Cloud with Vertex AI patterns.

The goal is not shallow AI literacy. The goal is to think, build, debug, and ship like the engineer companies trust to take AI from prototype to production.

## Learning Outcomes
By the end of this workshop, participants will be able to:
- Build AI agents that work beyond demos.
- Debug real production failure modes.
- Evaluate reliability using observability, evals, and human-in-the-loop checkpoints.
- Deploy to cloud-hosted infrastructure aligned with enterprise requirements.
- Explain trade-offs to stakeholders in terms of business impact and risk.

## Duration and Format
- Recommended duration: 2 to 3 days intensive, or 6 half-day sessions.
- Audience: Software engineers, solutions engineers, platform engineers, and AI-forward technical consultants.
- Prerequisites:
  - Python basics.
  - REST API familiarity.
  - Basic command-line workflow.
  - General understanding of cloud deployment concepts.

## What You Will Learn
### Build AI agents that work beyond the demo
- Develop agents with LLM APIs, tools, knowledge bases, and enterprise systems.
- Understand agent loops, tool calling, structured outputs, and guardrails.
- Build a LangGraph-style agent flow that answers from APIs and internal data.

### Think like a production AI engineer
- Reason through tokens, context windows, embeddings, RAG, and failure modes.
- Debug hallucinations, retrieval misses, tool-call errors, and brittle workflows.
- Evaluate agents with observability, eval suites, HITL checkpoints, and release gates.

### Deploy agents into real customer environments
- Move from local prototype to hosted endpoint on Google Cloud and Vertex AI.
- Understand deployment paths, Model Garden options, scaling, and cloud operations.
- Make explicit trade-offs across latency, cost, reliability, security, and customer impact.

## Workshop Agenda
### Module 1: AI Building Blocks
AI foundations for FDEs: how LLMs work, embeddings, RAG failure modes, agent patterns, tool calling, schemas, and structured outputs.

### Module 2: End-to-End Agent Development
End-to-end agent build: use agent graphs, real tools, knowledge bases, LLM APIs, evals, observability, guardrails, and HITL checkpoints.

### Module 3: Deploying Agents on GCP
Cloud deployment for AI agents: Vertex AI, Model Garden, GCP hosting patterns, production operations, latency, cost, and scaling trade-offs.

## Chapters
### Chapter 1: The FDE Mindset in AI Delivery
- Why companies hire FDEs for AI programs.
- Moving from prototype success to production accountability.

### Chapter 2: LLM Behavior and System Constraints
- Context windows, token budgets, and model limits.
- Practical implications for reliability and product UX.

### Chapter 3: Retrieval and RAG Failure Modes
- Embeddings and retrieval pipelines.
- Diagnosing low-recall and low-precision retrieval behavior.

### Chapter 4: Tool Calling and Agent Control Loops
- Planner-executor patterns.
- Deterministic interfaces through schema-bound tool inputs.

### Chapter 5: Building the End-to-End Agent
- Implementing retrieval, tool routing, and response synthesis.
- Adding fallback logic for partial failures.

### Chapter 6: Guardrails, Safety, and Human Approval
- Input/output policy checks.
- Human-in-the-loop checkpoints for high-risk actions.

### Chapter 7: Evaluation and Observability
- Defining eval cases and pass/fail criteria.
- Logging latency, token usage, confidence, and error classes.

### Chapter 8: Deployment on GCP and Vertex AI
- Hosting patterns and API architecture.
- Trade-offs across latency, cost, reliability, and security.

### Chapter 9: Production Incident Playbooks
- Failure triage flow for retrieval, tool, and model issues.
- Rollback, fallback, and communication patterns.

### Chapter 10: Capstone Build and Review
- Team delivery of a production-minded agent.
- Architecture review with risk and business-impact framing.

## Hands-On Assets in This Workshop
- Python implementation: code/fde_agent_workshop.py
- Notebook lab: notebooks/fde_agent_workshop.ipynb
- Instructor training material: workshop_training_material.md

## Chapter Content Files
- chapter1-fde-mindset-ai-delivery.md
- chapter2-llm-behavior-system-constraints.md
- chapter3-retrieval-rag-failure-modes.md
- chapter4-tool-calling-agent-control-loops.md
- chapter5-building-end-to-end-agent.md
- chapter6-guardrails-safety-human-approval.md
- chapter7-evaluation-observability.md
- chapter8-deployment-gcp-vertex-ai.md
- chapter9-production-incident-playbooks.md
- chapter10-capstone-build-review.md

## Chapter Code Files
- code/chapter1_fde_mindset_ai_delivery.py
- code/chapter2_llm_behavior_system_constraints.py
- code/chapter3_retrieval_rag_failure_modes.py
- code/chapter4_tool_calling_agent_control_loops.py
- code/chapter5_building_end_to_end_agent.py
- code/chapter6_guardrails_safety_human_approval.py
- code/chapter7_evaluation_observability.py
- code/chapter8_deployment_gcp_vertex_ai.py
- code/chapter9_production_incident_playbooks.py
- code/chapter10_capstone_build_review.py

## Chapter Slide Decks
- slides/chapter1-fde-mindset-slides.md
- slides/chapter2-llm-constraints-slides.md
- slides/chapter3-rag-failure-modes-slides.md
- slides/chapter4-tool-calling-control-loops-slides.md
- slides/chapter5-end-to-end-agent-slides.md
- slides/chapter6-guardrails-safety-hitl-slides.md
- slides/chapter7-evaluation-observability-slides.md
- slides/chapter8-deployment-gcp-vertex-slides.md
- slides/chapter9-incident-playbooks-slides.md
- slides/chapter10-capstone-review-slides.md

## Suggested Delivery Flow
1. Kickoff and architecture framing.
2. Module 1 with mini exercises.
3. Module 2 coding lab (agent + tools + retrieval + guardrails).
4. Module 3 deployment design and operations simulation.
5. Review, Q and A, and implementation roadmap.

## Expected Outputs
Participants should leave with:
- A working local agent prototype.
- A deployment-ready architecture sketch for GCP/Vertex AI.
- A failure-mode checklist and evaluation plan.
- A stakeholder-ready explanation of production trade-offs.
