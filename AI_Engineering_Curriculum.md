# AI Engineering Curriculum
### From Zero to Production-Ready AI Developer
*Three 8-Week Courses for University Students & Early Software Engineers*

---

> **Who This Is For:** Computer science/engineering students, bootcamp grads, or early-career software engineers (0–3 years) who know how to code in Python but have little to no prior AI/ML experience. Each course builds directly on the previous one.

---

## Curriculum Overview

| Course | Title | Focus | Capstone |
|--------|-------|--------|----------|
| **Course 1** | Foundations of AI Engineering | Prompting, APIs, RAG, Fine-Tuning | AI-Powered Study Companion |
| **Course 2** | Building AI Systems | MCP Servers, AI Agents, Orchestration | Autonomous Research Agent |
| **Course 3** | Production AI Engineering | Evals, Observability, Safety, Deployment | End-to-End AI Product |

---

---

# COURSE 1: Foundations of AI Engineering

**Duration:** 8 Weeks | **Level:** Beginner → Intermediate
**Prerequisites:** Python basics, comfort with REST APIs, basic command line usage

## Course Description

This course introduces the foundational concepts every AI engineer needs before building complex systems. You will go from understanding what LLMs actually are, to writing production-quality prompts, calling AI APIs in real applications, storing and retrieving knowledge with RAG pipelines, and customizing models through fine-tuning. By the end, you'll have the vocabulary, mental models, and hands-on code to tackle Course 2.

## Learning Objectives

By completing this course, students will be able to:
- Explain how large language models work at a conceptual and practical level
- Write effective prompts using structured techniques (zero-shot, few-shot, chain-of-thought)
- Build applications using the Mistral API as the primary SDK, with OpenAI and Anthropic as comparison examples
- Design and implement a Retrieval-Augmented Generation (RAG) pipeline from scratch
- Fine-tune a pre-trained model on a custom dataset
- Evaluate LLM outputs with basic quality metrics

---

## Weekly Breakdown

---

### Week 1: How LLMs Actually Work

**Theme:** Build intuition before writing code

#### Learning Goals
- Understand the transformer architecture at a conceptual level
- Know the difference between training, fine-tuning, and inference
- Understand tokenization, embeddings, and context windows
- Navigate the AI model landscape (OpenAI, Anthropic, Mistral, Llama, etc.)

#### Chapter 1.1 — The Transformer Demystified
- What is attention and why it changed everything
- Tokens, not words: how LLMs see text
- Parameters, weights, and what "training" actually means
- The context window: what it is, why it matters, how to think about it

#### Chapter 1.2 — The LLM Landscape
- Closed vs. open-source models: trade-offs you'll face as an engineer
- Model families and their strengths: Claude, GPT-4, Gemini, Llama, Mistral
- Temperature, top-p, and other sampling parameters — what they control
- Reading a model card: what to look for before picking a model

#### Chapter 1.3 — Your First API Call
- Setting up a Python environment for AI development
- Calling the Mistral API for core labs, with OpenAI and Anthropic comparison snippets
- Handling responses, errors, and rate limits
- Building a simple CLI chatbot in 50 lines of Python

#### What to Expect
- **Lecture time:** ~4 hours
- **Lab:** Build a CLI chatbot that streams responses
- **Reading:** Attention Is All You Need (summary version), Mistral model documentation, plus optional OpenAI/Anthropic API docs for comparison
- **Quiz:** 10-question concept check on transformers and tokenization

---

### Week 2: Prompt Engineering

**Theme:** Prompting is programming

#### Learning Goals
- Apply structured prompting techniques (zero-shot, few-shot, CoT, role prompting)
- Write system prompts that reliably control model behavior
- Debug bad outputs by diagnosing prompt failures
- Build reusable prompt templates

#### Chapter 2.1 — The Anatomy of a Prompt
- System vs. user vs. assistant roles
- Instruction clarity: why vague prompts fail
- Context injection: giving the model what it needs
- Output formatting: JSON, Markdown, structured schemas

#### Chapter 2.2 — Core Prompting Techniques
- Zero-shot prompting: asking directly
- Few-shot prompting: showing examples
- Chain-of-thought (CoT): forcing step-by-step reasoning
- Role prompting and persona assignment
- Self-consistency and prompt ensembling

#### Chapter 2.3 — Prompt Debugging and Iteration
- Common failure modes: hallucination, refusal, drift, format breakage
- A/B testing prompts: comparing outputs systematically
- Building a prompt versioning system
- When to change the prompt vs. change the model

#### What to Expect
- **Lecture time:** ~3.5 hours
- **Lab:** Build a prompt testing harness that compares 3 prompt variants side by side
- **Assignment:** Write a 5-prompt system for a customer support agent and evaluate it against 20 test cases
- **Deliverable:** Documented prompt library with version history

---

### Week 3: Working with AI APIs at Scale

**Theme:** From single calls to real applications

#### Learning Goals
- Architect multi-turn conversations with memory
- Handle streaming, retries, and error states
- Manage costs and token budgets
- Build a basic AI-powered web endpoint

#### Chapter 3.1 — Conversation State and Memory
- Stateless APIs: why you must send history every time
- Rolling window memory vs. summary memory vs. vector memory
- When context gets too long: truncation strategies
- Building a conversation manager class

#### Chapter 3.2 — Streaming and Async Patterns
- Streaming responses with `stream=True`
- Async API calls with `asyncio` and `httpx`
- Handling partial outputs gracefully in the UI
- Retry logic with exponential backoff

#### Chapter 3.3 — Cost Management and Optimization
- Token counting before sending requests
- Caching strategies: semantic caching and exact caching
- Batching requests for throughput
- Building a cost dashboard for your app

#### Chapter 3.4 — Building a FastAPI AI Endpoint
- Wrapping LLM calls in a REST API
- Streaming responses through HTTP Server-Sent Events (SSE)
- Input validation and sanitization
- Basic rate limiting for your AI endpoint

#### What to Expect
- **Lecture time:** ~4 hours
- **Lab:** Build a multi-turn AI assistant with a FastAPI backend and streaming support
- **Assignment:** Add cost tracking and a /stats endpoint to your API
- **Code Review:** Peer review of async and error-handling patterns

---

### Week 4: Embeddings and Vector Databases

**Theme:** Give your AI long-term memory

#### Learning Goals
- Understand what embeddings are and how semantic similarity works
- Generate embeddings using the OpenAI and Anthropic APIs
- Set up and query a vector database (Pinecone, Weaviate, or ChromaDB)
- Implement similarity search in a real application

#### Chapter 4.1 — What Are Embeddings?
- Vectors and high-dimensional space (without the math terror)
- Why "meaning" can be represented as numbers
- Embedding models: text-embedding-3, voyage-ai, Cohere
- Comparing embeddings: cosine similarity, dot product, Euclidean distance

#### Chapter 4.2 — Vector Databases
- Why you can't just use PostgreSQL for this (and when you can)
- ChromaDB: local, zero-config vector search for development
- Pinecone: managed, production-grade vector storage
- pgvector: Postgres extension for when SQL and vectors coexist
- Index types: HNSW vs. IVFFlat — when performance matters

#### Chapter 4.3 — Chunking and Indexing Documents
- Why you chunk: context windows and retrieval precision
- Fixed-size chunking vs. semantic chunking vs. recursive splitting
- Overlapping chunks and why you need them
- Metadata filtering: combining vector search with structured filters

#### What to Expect
- **Lecture time:** ~3.5 hours
- **Lab:** Embed 500 Wikipedia articles into ChromaDB and build a semantic search UI
- **Assignment:** Compare retrieval quality using 3 different chunking strategies
- **Quiz:** Concept check on embeddings and vector similarity

---

### Week 5: Retrieval-Augmented Generation (RAG)

**Theme:** Teach your AI to use external knowledge

#### Learning Goals
- Build a complete RAG pipeline end-to-end
- Implement retrieval, reranking, and context injection
- Evaluate retrieval quality with NDCG and hit rate metrics
- Architect RAG for production: indexing pipelines and refresh strategies

#### Chapter 5.1 — RAG Architecture Fundamentals
- Why RAG exists: the problem with relying on parametric memory
- The retrieval-augmented generation loop: retrieve → inject → generate
- Naive RAG vs. Advanced RAG vs. Modular RAG
- When to use RAG vs. fine-tuning vs. both

#### Chapter 5.2 — Building the Retrieval Pipeline
- Document loaders: PDFs, HTML, Notion, GitHub, Confluence
- Text splitters and chunking revisited with LangChain/LlamaIndex
- Embedding and upserting documents into your vector store
- Query transformation: HyDE, query expansion, sub-question decomposition

#### Chapter 5.3 — Reranking and Context Assembly
- Why raw retrieval isn't enough: the lost-in-the-middle problem
- Cross-encoder rerankers: Cohere Rerank, bge-reranker
- Context window packing: ordering retrieved chunks strategically
- Filtering irrelevant chunks before injecting

#### Chapter 5.4 — Evaluating RAG Quality
- Retrieval metrics: hit rate, MRR, NDCG@k
- Generation metrics: faithfulness, answer relevance, context recall
- Using RAGAS for automated evaluation
- Building a golden QA dataset for ongoing regression testing

#### What to Expect
- **Lecture time:** ~5 hours (biggest week)
- **Lab:** Build a RAG system over a corpus of technical documentation
- **Assignment:** Run RAGAS evaluations and write a 1-page report on retrieval quality
- **Deliverable:** Documented RAG system with evaluation report

---

### Week 6: Fine-Tuning Models

**Theme:** When prompting isn't enough

#### Learning Goals
- Understand the difference between pre-training, full fine-tuning, and PEFT
- Prepare a fine-tuning dataset from scratch
- Fine-tune a model using the OpenAI fine-tuning API and Hugging Face
- Evaluate fine-tuned models against baseline prompting

#### Chapter 6.1 — Fine-Tuning Concepts
- Parametric memory vs. contextual injection: what fine-tuning actually changes
- Full fine-tuning vs. PEFT: LoRA, QLoRA, adapters
- When fine-tuning is worth it (and when it's not)
- Dataset size rules of thumb: quality over quantity

#### Chapter 6.2 — Preparing Your Dataset
- Supervised fine-tuning format: instruction → response pairs
- Cleaning and deduplicating training data
- Handling class imbalance in fine-tuning datasets
- Generating synthetic training data with Claude

#### Chapter 6.3 — Fine-Tuning in Practice
- Fine-tuning GPT-4o-mini via the OpenAI API (practical and affordable)
- Fine-tuning Llama 3 with QLoRA on Google Colab
- Hyperparameters: epochs, learning rate, batch size
- Checkpointing, early stopping, and overfitting signals

#### Chapter 6.4 — Evaluating Fine-Tuned Models
- Side-by-side comparison: base model vs. fine-tuned model
- Task-specific metrics: BLEU, ROUGE, F1 for classification
- Human evaluation rubrics
- Cost-performance trade-off analysis

#### What to Expect
- **Lecture time:** ~4 hours
- **Lab:** Fine-tune GPT-4o-mini on a domain-specific Q&A dataset
- **Assignment:** Compare your fine-tuned model to a well-prompted base model on 50 test cases
- **Deliverable:** Fine-tuning report with performance comparison

---

### Week 7: Introduction to Evaluation Systems

**Theme:** If you can't measure it, you can't improve it

#### Learning Goals
- Build a systematic evaluation harness for LLM outputs
- Implement LLM-as-judge evaluation patterns
- Understand human evaluation vs. automated evaluation trade-offs
- Track model performance over time

#### Chapter 7.1 — Why Evals Matter
- The vibe check problem: why eyeballing outputs doesn't scale
- Offline evals vs. online evals vs. human evals
- Building your golden dataset: what makes a good test case
- Regression testing: making sure improvements don't break other things

#### Chapter 7.2 — Automated Evaluation Techniques
- LLM-as-judge: using Claude to grade Claude
- Rubric-based evaluation prompts
- Reference-based vs. reference-free evaluation
- G-Eval and other structured evaluation frameworks

#### Chapter 7.3 — Building an Evaluation Pipeline
- Structuring an eval dataset as JSONL
- Running evals at scale with async API calls
- Storing and visualizing eval results
- CI integration: running evals on every pull request

#### What to Expect
- **Lecture time:** ~3 hours
- **Lab:** Build an eval harness for your RAG system from Week 5
- **Assignment:** Write 30 golden test cases for your Study Companion capstone project

---

### Week 8: Capstone Project — AI-Powered Study Companion

**Theme:** Bring it all together

#### Project Description
Build a fully functional AI-powered study companion that helps university students learn any subject by ingesting course materials and providing personalized tutoring, quiz generation, and concept explanation.

#### Capstone Requirements

**Core Features (Required)**
- Document ingestion pipeline supporting PDFs and plain text
- RAG system that answers questions from uploaded course materials
- Multi-turn conversation with persistent session memory
- Concept explanation with adaptive depth (explain like I'm 5 → explain technically)
- Quiz generation mode: generates practice questions from ingested material

**Stretch Goals (Choose 1)**
- Fine-tune a small model on educational Q&A pairs for better pedagogical tone
- Add citation tracking: every answer cites the exact source chunk
- Build a simple React or Streamlit frontend
- Add evaluation tooling that scores the tutor's answer quality

#### Deliverables
1. **GitHub repository** with clean, documented code
2. **Architecture diagram** showing data flow from document upload to answer generation
3. **Evaluation report** — run your eval harness against 30 test cases and report results
4. **5-minute demo video** or live demo presentation
5. **README** with setup instructions, architectural decisions, and lessons learned

#### Grading Rubric
| Component | Weight |
|-----------|--------|
| RAG pipeline functionality | 30% |
| Prompt quality and conversation design | 20% |
| Evaluation harness and results | 20% |
| Code quality and documentation | 15% |
| Demo clarity and architectural reasoning | 15% |

---

---

# COURSE 2: Building AI Systems

**Duration:** 8 Weeks | **Level:** Intermediate
**Prerequisites:** Course 1 completed, or equivalent experience with LLM APIs, RAG, and embeddings

## Course Description

This course moves from single-model applications to interconnected AI systems. You'll learn how to give AI models access to external tools, build autonomous agents that can plan and execute multi-step tasks, design multi-agent workflows where specialized agents collaborate, and integrate with the broader ecosystem through Model Context Protocol (MCP) servers. By the end, you'll be able to design and build systems that can act in the world — not just answer questions.

## Learning Objectives

By completing this course, students will be able to:
- Build MCP servers that expose tools and data to AI models
- Design and implement AI agents using planning and tool-use patterns
- Architect multi-agent systems with specialized roles and communication protocols
- Apply orchestration patterns: supervisor, pipeline, swarm, and hierarchical agents
- Understand and mitigate risks in agentic systems (prompt injection, loop detection, scope creep)
- Build an autonomous research agent as a capstone project

---

## Weekly Breakdown

---

### Week 1: Tool Use and Function Calling

**Theme:** Give your AI hands

#### Learning Goals
- Implement function calling with the Mistral API, with OpenAI and Anthropic as comparison examples
- Design tool schemas that are clear and unambiguous for models
- Handle tool call loops, errors, and partial execution
- Build a tool registry pattern for reusable tool libraries

#### Chapter 1.1 — How Function Calling Works
- From text completion to action: the function calling paradigm
- Tool schemas: name, description, parameters — what the model sees
- The tool call loop: LLM calls tool → tool returns result → LLM continues
- Parallel tool calls: when models call multiple tools at once

#### Chapter 1.2 — Designing Good Tool Schemas
- Writing descriptions that guide model behavior, not just document it
- Parameter design: required vs. optional, enums, nested objects
- Naming conventions that prevent model confusion
- Error returns: how to communicate failure back to the model

#### Chapter 1.3 — Tool Implementation Patterns
- Building a tool executor class
- Sandboxing tool execution: why you don't just call user-supplied functions
- Timeout and error handling in tool execution
- Logging tool calls for debugging and auditing

#### Chapter 1.4 — Real Tools: Web, Files, Code
- Web search tool: using SerpAPI or Brave Search
- File system tools: read, write, list with safety constraints
- Code execution tool: running Python in a sandbox (E2B, subprocess)
- Calculator and structured data tools

#### What to Expect
- **Lecture time:** ~4 hours
- **Lab:** Build a personal assistant with 5 tools: web search, calculator, file read/write, and current time
- **Assignment:** Add error handling and retry logic; log all tool calls to SQLite

---

### Week 2: The Model Context Protocol (MCP)

**Theme:** The USB-C standard for AI tool integration

#### Learning Goals
- Understand the MCP specification and why it exists
- Build an MCP server that exposes custom tools and resources
- Connect MCP servers to Claude and other clients
- Design MCP servers for real-world data sources

#### Chapter 2.1 — What Is MCP and Why Does It Matter?
- The fragmentation problem: every AI app reinventing tool integration
- MCP as a universal protocol: servers, clients, and hosts
- MCP primitives: tools, resources, prompts, sampling
- The MCP ecosystem: Claude Desktop, IDEs, and custom clients

#### Chapter 2.2 — Building Your First MCP Server
- MCP server anatomy: TypeScript SDK vs. Python SDK
- Defining tools with JSON Schema
- Implementing tool handlers
- Running and testing your MCP server locally

#### Chapter 2.3 — Resources and Dynamic Context
- Resources vs. tools: when to use each
- Exposing file systems, databases, and APIs as MCP resources
- URI patterns for resource addressing
- Streaming large resources

#### Chapter 2.4 — Real-World MCP Server Patterns
- GitHub MCP server: exposing repo data to Claude
- Database MCP server: SQL query interface over Postgres
- Internal knowledge base MCP server: company documentation
- Testing your MCP server with the MCP Inspector

#### What to Expect
- **Lecture time:** ~4 hours
- **Lab:** Build an MCP server for a GitHub repository that exposes: file tree, file contents, open issues, and PR summaries
- **Assignment:** Extend the server to support write operations (create issue, comment on PR) with safety guardrails

---

### Week 3: AI Agent Foundations

**Theme:** From single calls to autonomous loops

#### Learning Goals
- Define what makes something an "agent" vs. a chain vs. a workflow
- Implement a ReAct (Reason + Act) agent from scratch
- Handle agent loops: stopping conditions, max iterations, progress detection
- Understand the planning-execution-reflection cycle

#### Chapter 3.1 — What Is an Agent?
- The spectrum: LLM → Chain → Agent → Multi-Agent
- Core properties: perception, planning, action, memory
- The agency paradox: more autonomy = more power = more risk
- Agentic vs. deterministic: when to use which

#### Chapter 3.2 — The ReAct Pattern
- Reason → Act → Observe → Repeat: the core loop
- Implementing ReAct with tool use
- Thought traces: making agent reasoning visible
- Stopping conditions: success, failure, and "I don't know"

#### Chapter 3.3 — Agent Memory Architecture
- In-context memory: conversation history as working memory
- External memory: vector DB for episodic recall
- Procedural memory: learned strategies stored as prompts
- Memory write and retrieval strategies

#### Chapter 3.4 — Agent Reliability Patterns
- Loop detection: recognizing when an agent is spinning
- Progress signals: detecting forward movement
- Fallback strategies: when to give up and ask for help
- Human-in-the-loop: when and how to pause for approval

#### What to Expect
- **Lecture time:** ~4 hours
- **Lab:** Build a ReAct agent that can answer complex research questions using web search, Wikipedia, and a calculator
- **Assignment:** Add loop detection, a max-iteration limit, and a confidence score for the final answer

---

### Week 4: Planning and Complex Reasoning

**Theme:** Agents that think before they act

#### Learning Goals
- Implement planning architectures: plan-and-execute, tree-of-thought, graph-of-thought
- Build agents that decompose complex tasks into subtasks
- Handle plan revision when execution fails
- Implement reflection and self-critique loops

#### Chapter 4.1 — Planning Architectures
- Plan-and-execute vs. ReAct: planning upfront vs. interleaved
- Tree of Thought: branching reasoning paths with backtracking
- Graph of Thought: non-linear reasoning structures
- When planning overhead is worth it vs. when to just act

#### Chapter 4.2 — Task Decomposition
- Hierarchical task decomposition: breaking goals into subtasks
- Dependency graphs: which tasks block which
- Dynamic replanning: updating the plan as new information arrives
- The meta-planner pattern: a planner that plans the planning

#### Chapter 4.3 — Reflection and Self-Critique
- Reflexion: using failure to improve future attempts
- Self-critique loops: asking the model to grade its own output
- Constitutional AI principles applied to agent self-correction
- When reflection improves outcomes vs. when it causes overthinking

#### What to Expect
- **Lecture time:** ~4 hours
- **Lab:** Build a plan-and-execute agent that can complete multi-step tasks like "research, outline, and draft a 500-word blog post on topic X"
- **Assignment:** Add reflection — after completing the task, the agent critiques its own work and proposes one improvement

---

### Week 5: Multi-Agent Systems

**Theme:** Divide and conquer with specialized agents

#### Learning Goals
- Design multi-agent architectures: supervisor, pipeline, swarm
- Implement agent-to-agent communication patterns
- Assign specialized roles and route tasks appropriately
- Handle failures in one agent without crashing the whole system

#### Chapter 5.1 — Multi-Agent Architecture Patterns
- Supervisor-worker: a coordinator delegates to specialist agents
- Sequential pipeline: output of agent A becomes input to agent B
- Parallel fan-out: multiple agents tackle different aspects simultaneously
- Swarm: decentralized agents with emergent coordination

#### Chapter 5.2 — Agent Communication
- Structured message passing between agents
- Shared state: a scratchpad both agents can read/write
- Event-driven architectures: agents reacting to each other's outputs
- Conversation handoffs: passing the full context when routing

#### Chapter 5.3 — Specialization and Routing
- When specialization helps: research agent vs. writing agent vs. code agent
- Router agents: using an LLM to classify and route incoming tasks
- Capability registries: agents advertising what they can do
- Fallback chains: what happens when the specialist fails

#### Chapter 5.4 — Resilience in Multi-Agent Systems
- Fault isolation: one agent failing shouldn't cascade
- Retry and compensation patterns
- Monitoring agent health in a running system
- Dead letter queues for tasks no agent could complete

#### What to Expect
- **Lecture time:** ~4.5 hours
- **Lab:** Build a 3-agent research system: Router → Researcher + Writer → Editor
- **Assignment:** Add a monitoring dashboard showing each agent's task count, success rate, and average latency

---

### Week 6: Agent Safety and Security

**Theme:** Powerful systems need guardrails

#### Learning Goals
- Identify and mitigate prompt injection attacks in agentic systems
- Implement permission scoping and capability restrictions
- Build human-in-the-loop checkpoints for high-risk actions
- Design audit logs for agent actions

#### Chapter 6.1 — Threat Modeling Agentic Systems
- What goes wrong when agents have too much power
- Prompt injection: malicious content hijacking agent behavior
- Scope creep: agents doing things they weren't supposed to
- Catastrophic actions: deleting files, sending emails, making purchases

#### Chapter 6.2 — Prompt Injection Defense
- Injection anatomy: how malicious instructions sneak in
- Input sanitization for external content
- Instruction hierarchy: system prompt authority over tool output
- Detecting injection attempts with classifier models

#### Chapter 6.3 — Permission Systems and Sandboxing
- Principle of least privilege for agents
- Capability lists: explicitly enumerating what an agent can do
- Sandboxing code execution with E2B or Daytona
- Read-only vs. read-write modes with confirmation gates

#### Chapter 6.4 — Human-in-the-Loop Patterns
- Identifying which actions require human approval
- Async approval flows: Slack notifications, email, UI widgets
- Reversible vs. irreversible actions: different thresholds
- Audit logs: what to record for accountability

#### What to Expect
- **Lecture time:** ~4 hours
- **Lab:** Red-team your Week 5 multi-agent system — find 3 ways to make it misbehave, then patch each one
- **Assignment:** Implement a permission system and audit log; write a 1-page threat model for your capstone project

---

### Week 7: Agent Orchestration Frameworks

**Theme:** Stand on the shoulders of giants

#### Learning Goals
- Navigate the agent framework landscape (LangGraph, CrewAI, AutoGen, custom)
- Build a workflow in LangGraph with conditional branching
- Understand when frameworks help vs. when they add complexity
- Migrate a hand-built agent to a framework

#### Chapter 7.1 — The Framework Landscape
- Why frameworks exist: common patterns, boilerplate reduction
- LangGraph: stateful graph-based workflows with cycles
- CrewAI: role-based multi-agent collaboration
- AutoGen: conversational multi-agent with Microsoft Research roots
- The "no framework" argument: when raw API calls are cleaner

#### Chapter 7.2 — LangGraph Deep Dive
- Nodes, edges, and conditional routing
- State management across graph traversal
- Cycles: building loops that don't spin forever
- Checkpointing: resuming interrupted workflows

#### Chapter 7.3 — Choosing the Right Tool
- Framework decision matrix: complexity, team familiarity, debugging ease
- Escape hatches: mixing framework code with raw API calls
- Observability: which frameworks make debugging easier
- Migration paths: from prototype to production

#### What to Expect
- **Lecture time:** ~3.5 hours
- **Lab:** Rebuild your Week 3 ReAct agent in LangGraph; compare the development experience
- **Assignment:** Write a 500-word reflection: what did the framework make easier, what did it obscure?

---

### Week 8: Capstone Project — Autonomous Research Agent

**Theme:** Build something that can work while you sleep

#### Project Description
Build an autonomous research agent that can take a high-level research question, break it down into sub-questions, search for answers across multiple sources, synthesize findings, detect contradictions, and produce a structured research report — all without human intervention after the initial prompt.

#### Capstone Requirements

**Core Features (Required)**
- Accepts a research question and optional constraints (time limit, source types, depth)
- Decomposes the question into 3–7 sub-questions using a planner agent
- Researcher agent uses web search, Wikipedia, and arxiv tools to gather evidence
- Writer agent synthesizes findings into a structured Markdown report with citations
- Contradiction detector flags conflicting claims in the gathered sources
- Human-in-the-loop checkpoint: shows plan before execution, asks for approval

**Stretch Goals (Choose 1)**
- Multi-agent parallelism: run researcher tasks for each sub-question concurrently
- MCP server: expose your research agent as an MCP tool for Claude Desktop
- Evaluation harness: score report quality on factual accuracy and citation coverage
- Persistent memory: the agent remembers prior research sessions and avoids re-searching

#### Deliverables
1. **GitHub repository** with full source code and clean README
2. **Architecture diagram** showing agent roles, communication flows, and tool integrations
3. **3 sample research reports** generated by the system on topics of your choice
4. **Security audit doc** — what are the 3 biggest risks in your design and how did you mitigate them?
5. **Demo:** live or recorded, showing end-to-end from question to report

#### Grading Rubric
| Component | Weight |
|-----------|--------|
| Agent architecture and reasoning quality | 30% |
| Tool integration and MCP usage | 20% |
| Safety and human-in-the-loop design | 20% |
| Code quality, documentation, observability | 15% |
| Demo and architectural reasoning | 15% |

---

---

# COURSE 3: Production AI Engineering

**Duration:** 8 Weeks | **Level:** Intermediate → Advanced
**Prerequisites:** Courses 1 & 2 completed, or strong experience building AI agents and RAG systems

## Course Description

Building AI systems that work in a demo is table stakes. This course is about what comes after: evaluation frameworks that catch regressions before users do, observability that tells you why something failed at 2am, safety systems that keep your AI behaving correctly at scale, and deployment patterns that don't break under real traffic. You'll also explore the frontier topics that will define AI engineering in the next 2–3 years: multimodal systems, structured generation, and the evolving role of the AI engineer. The final capstone asks you to ship a complete, production-ready AI product.

## Learning Objectives

By completing this course, students will be able to:
- Design and implement comprehensive evaluation systems for LLM applications
- Build observability infrastructure: tracing, logging, and alerting for AI systems
- Apply safety engineering practices: content moderation, output validation, jailbreak detection
- Deploy AI applications with proper scaling, caching, and cost controls
- Work with multimodal inputs (text, images, documents)
- Use structured generation for reliable, schema-constrained outputs
- Ship a complete AI product as a final capstone

---

## Weekly Breakdown

---

### Week 1: Evaluation Systems at Scale

**Theme:** Systematic quality assurance for AI

#### Learning Goals
- Design a multi-dimensional evaluation framework
- Implement LLM-as-judge at scale with consistency controls
- Build regression testing pipelines for AI systems
- Integrate evals into CI/CD workflows

#### Chapter 1.1 — Evaluation Framework Design
- The eval pyramid: unit evals → integration evals → end-to-end evals
- Dimensions of quality: correctness, coherence, faithfulness, safety, tone
- Building eval personas: different user types with different quality expectations
- The golden dataset lifecycle: creation, curation, expansion, retirement

#### Chapter 1.2 — LLM-as-Judge at Scale
- Judge model selection: why you need a stronger model to judge a weaker one
- Positional bias, verbosity bias, self-preference bias — and how to counter them
- Structured judge rubrics with 1–5 scales and explicit criteria
- Inter-rater reliability: comparing LLM judge to human judge agreement

#### Chapter 1.3 — Evaluation Frameworks and Tools
- RAGAS for RAG evaluation: faithfulness, answer relevance, context precision
- PromptFoo: LLM testing with YAML-defined test suites
- Braintrust: eval tracking, dataset management, and regression detection
- Building a custom eval runner with async parallelism

#### Chapter 1.4 — Evals in CI/CD
- GitHub Actions workflow for running evals on every PR
- Regression thresholds: when a PR blocks deployment
- Eval result dashboards: tracking quality trends over time
- The eval debt problem: what happens when you skip writing evals

#### What to Expect
- **Lecture time:** ~4 hours
- **Lab:** Build a RAGAS-powered eval pipeline for a RAG system; integrate it into a GitHub Actions workflow
- **Assignment:** Write 50 golden test cases for your capstone project, covering at least 5 quality dimensions

---

### Week 2: Observability and Debugging

**Theme:** See what your AI is actually doing

#### Learning Goals
- Implement distributed tracing for multi-step AI pipelines
- Build structured logging that captures model inputs, outputs, and metadata
- Set up alerting for quality degradation and cost spikes
- Debug AI failures using traces and logs

#### Chapter 2.1 — AI Observability Fundamentals
- Why traditional APM isn't enough for AI systems
- Traces, spans, and events: the observability data model applied to LLMs
- What to log: inputs, outputs, latency, token counts, costs, tool calls
- The feedback loop: capturing user signals (thumbs up/down, edits, abandonment)

#### Chapter 2.2 — Tracing AI Pipelines
- OpenTelemetry for AI: instrumenting LLM calls
- LangSmith: tracing for LangChain-based systems
- Arize Phoenix: open-source AI observability
- Braintrust Logging: production trace collection and analysis

#### Chapter 2.3 — Dashboards and Alerting
- Key metrics: p50/p95/p99 latency, error rate, cost per request, quality score
- Building dashboards in Grafana or Datadog
- Alert thresholds for: quality score drop, cost spike, error rate increase
- On-call playbooks: what to check when an alert fires at 2am

#### Chapter 2.4 — Debugging Failure Modes
- Anatomy of an AI incident: from symptom to root cause
- Token budget failures: context too long, important content truncated
- Retrieval failures: wrong chunks, empty results, irrelevant context
- Model behavior drift: same prompt, different output across model versions

#### What to Expect
- **Lecture time:** ~4 hours
- **Lab:** Instrument a multi-agent system with OpenTelemetry; build a Grafana dashboard for it
- **Assignment:** Inject 5 deliberate failures into a RAG system; use your traces to diagnose each one

---

### Week 3: Structured Generation and Output Reliability

**Theme:** Make your AI output machine-parseable, every time

#### Learning Goals
- Use structured generation to guarantee schema-compliant outputs
- Implement validation and repair loops for LLM outputs
- Apply constrained decoding with Outlines and similar tools
- Design output schemas that guide model behavior

#### Chapter 3.1 — The Output Reliability Problem
- Why `json.loads(llm_output)` fails in production
- Parsing failures: partial JSON, incorrect types, missing fields
- The prompt-vs-constraint trade-off: asking nicely vs. enforcing structurally

#### Chapter 3.2 — Structured Generation Techniques
- Structured output and `response_format` with the Mistral API
- OpenAI JSON mode and Anthropic tool use as comparison patterns
- Pydantic + Instructor: type-safe LLM outputs in Python
- Outlines: constrained decoding at the token level for open-source models

#### Chapter 3.3 — Schema Design for LLMs
- Designing schemas that guide reasoning, not just structure output
- Field descriptions as model instructions
- Union types, optionals, and discriminated unions — and when models struggle
- Schema versioning: handling model behavior changes without breaking downstream

#### Chapter 3.4 — Validation and Repair
- Multi-layer validation: syntax → semantics → business rules
- Repair prompts: asking the model to fix its own malformed output
- Fallback chains: structured → unstructured → human review
- Testing schema adherence across model versions

#### What to Expect
- **Lecture time:** ~3.5 hours
- **Lab:** Build a document extraction pipeline using Instructor that extracts structured data from unstructured PDFs with 99%+ schema compliance
- **Assignment:** Add a repair loop and validation cascade; measure how many requests require each fallback level

---

### Week 4: Safety Engineering

**Theme:** AI that behaves correctly, even when users don't

#### Learning Goals
- Implement content moderation pipelines using classifiers and LLM judges
- Build jailbreak and prompt injection detection
- Apply output validation for harmful or off-topic content
- Design safety systems that don't over-refuse legitimate requests

#### Chapter 4.1 — The Safety Engineering Landscape
- Harmful outputs: misinformation, dangerous instructions, toxic content
- Misuse patterns: jailbreaking, prompt injection, adversarial inputs
- Over-refusal: when safety systems damage the user experience
- Balancing safety and utility: the false positive/false negative trade-off

#### Chapter 4.2 — Input Safety
- Prompt injection detection: classifiers and heuristic rules
- PII detection and redaction before sending to external APIs
- Toxic input classification: Perspective API, OpenAI Moderation, custom classifiers
- Rate limiting and abuse detection at the API layer

#### Chapter 4.3 — Output Safety
- Output moderation: passing responses through a safety classifier
- Groundedness checking: detecting hallucinations and unsupported claims
- Factual consistency verification for high-stakes applications
- Safe-by-design prompts: constitutional prompting and system prompt guardrails

#### Chapter 4.4 — Red Teaming Your System
- Adversarial testing methodology: role-play attacks, indirect injection, jailbreak templates
- Automated red teaming tools: PyRIT, Garak
- Building a safety eval dataset from real-world attack patterns
- Incident response: what to do when a safety failure reaches production

#### What to Expect
- **Lecture time:** ~4 hours
- **Lab:** Red-team your capstone project with 20 adversarial inputs, then build defenses for each attack category
- **Assignment:** Write a safety specification document for your capstone: what content is in scope, out of scope, and how each is handled

---

### Week 5: Deployment and Scaling

**Theme:** From working code to production system

#### Learning Goals
- Deploy an AI application with Docker and a cloud provider
- Implement caching strategies to reduce latency and cost
- Design for scale: load balancing, autoscaling, and queue-based processing
- Manage model versions and rollback safely

#### Chapter 5.1 — Deployment Architecture for AI Apps
- Containerizing AI applications with Docker
- API gateway patterns: rate limiting, authentication, routing
- Stateless vs. stateful backends: implications for AI apps with memory
- Environment management: dev → staging → production

#### Chapter 5.2 — Caching and Cost Optimization
- Semantic caching: caching by meaning, not exact string match
- Prompt caching and request reuse patterns across Mistral, OpenAI, and Anthropic APIs
- CDN caching for static AI outputs (embeddings, pre-computed responses)
- Cost budgets, alerts, and per-tenant billing

#### Chapter 5.3 — Scaling Patterns
- Horizontal scaling: stateless API replicas behind a load balancer
- Queue-based processing: SQS/Redis for long-running agent tasks
- Model routing: sending simple queries to cheap models, complex to expensive
- Streaming at scale: SSE and WebSocket patterns under load

#### Chapter 5.4 — Model Version Management
- Pinning model versions in production
- Canary deployments: routing 5% of traffic to a new model
- Rollback triggers: automated and manual
- Migration testing: ensuring prompt compatibility across model versions

#### What to Expect
- **Lecture time:** ~4 hours
- **Lab:** Deploy your capstone project to Fly.io or Railway with Docker; add semantic caching with Redis
- **Assignment:** Load test your deployment with Locust; identify and fix your first bottleneck

---

### Week 6: Multimodal AI Engineering

**Theme:** AI that can see, read, and listen

#### Learning Goals
- Process images, PDFs, and documents with vision-capable models
- Build multimodal RAG pipelines
- Apply vision models to real-world document processing tasks
- Understand audio and video AI capabilities and their integration patterns

#### Chapter 6.1 — Vision Models in Production
- How vision-language models work: CLIP, LLaVA, Claude Vision, GPT-4V
- Image encoding: base64, URLs, file uploads
- Prompting for visual tasks: description, extraction, comparison, OCR
- Cost and latency implications of image inputs

#### Chapter 6.2 — Document Intelligence
- PDF processing: text extraction vs. visual processing trade-offs
- Table extraction from documents: structure parsing challenges
- Form processing: extracting key-value pairs from unstructured forms
- Building a multimodal document ingestion pipeline

#### Chapter 6.3 — Multimodal RAG
- Embedding images alongside text in a unified vector space
- Colpali: document retrieval using visual embeddings
- Late interaction models for multimodal retrieval
- Storing and serving image chunks with metadata

#### Chapter 6.4 — Audio and Video (Survey)
- Whisper and speech-to-text integration patterns
- Video understanding: frame sampling and multimodal context
- Real-time audio pipelines: latency constraints and chunking strategies
- When to use specialized models vs. general-purpose multimodal models

#### What to Expect
- **Lecture time:** ~4 hours
- **Lab:** Build a multimodal document Q&A system that handles PDFs with embedded charts, tables, and images
- **Assignment:** Process 10 real-world documents from different domains; report on extraction accuracy and failure modes

---

### Week 7: The AI Engineering Career and Field

**Theme:** Where you fit in this field and where it's going

#### Learning Goals
- Map the AI engineering career landscape and role definitions
- Understand the state of AI tooling, frameworks, and their evolution
- Engage with open research questions in AI systems
- Define your personal AI engineering focus area

#### Chapter 7.1 — Role Definitions in AI Engineering
- AI Engineer vs. ML Engineer vs. Data Scientist vs. AI Researcher
- What AI engineers actually do day-to-day at startups vs. big tech
- The full stack of AI work: from data pipelines to UX design
- Salary benchmarks, hiring signals, and portfolio expectations

#### Chapter 7.2 — The Evolving Tooling Landscape
- Framework consolidation: which tools are winning and why
- The context engineering shift: beyond prompt engineering
- Model capability growth: what gets easier as models improve
- Open-source vs. proprietary: the ongoing split

#### Chapter 7.3 — Open Problems Worth Knowing
- Long-context reasoning: models that truly understand 100k tokens
- Agentic reliability: getting agents to 99.9% task completion
- Evaluation: building evals that actually track what matters
- Compute efficiency: doing more with smaller, faster models

#### Chapter 7.4 — Building in Public and Career Development
- Open-source projects that hire their contributors
- Writing technical blog posts that get you noticed
- Conference talks and communities: where AI engineers gather
- GitHub portfolio strategy for AI engineers

#### What to Expect
- **Lecture time:** ~3 hours (lighter week to allow capstone sprint)
- **Guest Talk:** Invited AI engineer from industry (startup or big tech)
- **Assignment:** Write a 1-page personal learning plan: what's your AI engineering focus area, and what are your next 3 learning goals after this curriculum?

---

### Week 8: Capstone Project — End-to-End AI Product

**Theme:** Ship something you'd put in your portfolio

#### Project Description
Build and deploy a complete, production-ready AI product of your own design. This is an open-ended project — you define the product, the users, and the problem it solves. The capstone must demonstrate mastery across evaluation, observability, safety, and deployment, not just a clever use of an LLM.

#### Capstone Requirements

**Production Criteria (All Required)**
- Deployed to a public URL (Fly.io, Railway, Vercel, Render, or equivalent)
- Comprehensive evaluation suite with at least 50 golden test cases
- Observability: request tracing, cost tracking, and a quality dashboard
- Safety layer: input moderation, output validation, and documented threat model
- Semantic caching and at least one cost optimization technique
- Full README with architecture diagram, setup instructions, and lessons learned

**Product Criteria (All Required)**
- Solves a real problem for a clearly defined user
- Uses at least 2 of: RAG, agents, MCP, fine-tuning, multimodal, structured generation
- Multi-turn conversation or multi-step agentic workflow (not just a single LLM call)
- A simple but functional UI (Streamlit, Next.js, React, Gradio — your choice)

**Example Ideas (Pick One or Propose Your Own)**
- AI code reviewer that gives structured feedback on GitHub PRs
- Legal document summarizer with citation tracking and hallucination detection
- AI tutoring system for a specific subject with adaptive difficulty
- Research assistant for a specialized domain (medical literature, patent search, etc.)
- AI-powered test engineer that writes Playwright tests for web apps

#### Deliverables
1. **Live deployed application** with public URL
2. **GitHub repository** with complete source code
3. **Architecture writeup** — 1–2 pages covering system design decisions and trade-offs
4. **Eval report** — results of your evaluation suite, including at least 3 quality dimensions
5. **Observability screenshots** — dashboard showing real production traffic
6. **10-minute demo** — live product walkthrough and architecture explanation

#### Grading Rubric
| Component | Weight |
|-----------|--------|
| Product quality and real-world usefulness | 25% |
| Evaluation system depth and coverage | 20% |
| Production deployment and observability | 20% |
| Safety design and threat model | 15% |
| Code quality, documentation, architecture | 10% |
| Demo clarity and articulation of decisions | 10% |

---

---

## Recommended Tools and Technologies

### Development Environment
- **Python 3.11+** — primary language throughout the curriculum
- **Node.js 20+** — for MCP server development
- **Git + GitHub** — version control and portfolio
- **VS Code** + Claude extension — primary IDE

### AI APIs and Models
- **Mistral API** — primary LLM throughout the curriculum
- **OpenAI API** — for fine-tuning and comparison
- **Anthropic Claude API** — additional comparison examples for API design and tool use
- **Voyage AI or Cohere** — embedding models

### Data and Storage
- **ChromaDB** — local vector database for development
- **Pinecone** — managed vector database for production
- **Redis** — semantic caching and session state
- **PostgreSQL + pgvector** — relational + vector in one

### Frameworks and Libraries
- **LangChain / LangGraph** — orchestration and agentic workflows
- **Instructor** — structured outputs with Pydantic
- **FastAPI** — AI backend APIs
- **Streamlit / Gradio** — rapid AI UI prototyping

### Evaluation and Observability
- **RAGAS** — RAG evaluation
- **PromptFoo** — prompt testing
- **Braintrust** — eval tracking and logging
- **Arize Phoenix** — open-source AI observability

### Deployment
- **Docker** — containerization
- **Fly.io or Railway** — cloud deployment for students
- **GitHub Actions** — CI/CD with eval gates

---

## Suggested Learning Path

```
Course 1 Weeks 1–4    → Foundations (APIs, Prompting, Embeddings)
Course 1 Weeks 5–6    → RAG + Fine-Tuning
Course 1 Week 7–8     → Intro Evals + Capstone 1

        ↓

Course 2 Weeks 1–2    → Tool Use + MCP Servers
Course 2 Weeks 3–5    → Agents + Multi-Agent Systems
Course 2 Weeks 6–8    → Safety + Orchestration + Capstone 2

        ↓

Course 3 Weeks 1–3    → Evals at Scale + Observability + Structured Gen
Course 3 Weeks 4–6    → Safety Engineering + Deployment + Multimodal
Course 3 Weeks 7–8    → Career + Capstone 3
```

---

## Assessment Summary Across All Three Courses

| Assessment Type | Course 1 | Course 2 | Course 3 |
|----------------|----------|----------|----------|
| Weekly labs | 7 labs | 7 labs | 7 labs |
| Weekly assignments | 7 assignments | 7 assignments | 7 assignments |
| Concept quizzes | 3 quizzes | 2 quizzes | 2 quizzes |
| Capstone project | 1 major | 1 major | 1 major |
| Peer code review | 1 session | 1 session | 2 sessions |

---

*Curriculum version 1.0 — Designed for the 2025–2026 academic year. The AI engineering field moves fast; instructors should review and update tool recommendations each semester.*
