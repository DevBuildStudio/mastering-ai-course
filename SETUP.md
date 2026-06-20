# AI Engineering Curriculum — Setup Guide

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Node.js 20+ (for MCP servers)
- Git
- VS Code with Jupyter extension

### 2. Get Your Mistral API Key

1. Go to [console.mistral.ai](https://console.mistral.ai)
2. Sign up or log in to your Mistral account
3. Navigate to **API Keys** in the left sidebar
4. Click **Create new key**, give it a name (e.g. `ai-engineering-course`)
5. Copy the key immediately — it will not be shown again
6. Add billing info if prompted (free tier available for development)

> Keep your API key secret. Never commit it to git. Always load it from environment variables.

### 3. Environment Setup

```bash
# Clone or navigate to the courses directory
cd d:/gith/courses

# Create virtual environment (recommended: one per course)
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows

# Install Course 1 dependencies
pip install -r course1/requirements.txt

# Copy environment template
cp course1/.env.example course1/.env
# Edit course1/.env and add your MISTRAL_API_KEY
```

Each course has its own `requirements.txt`. To switch courses:

```bash
pip install -r course2/requirements.txt
pip install -r course3/requirements.txt
```

Your `.env` file should contain at minimum:

```
MISTRAL_API_KEY=your_key_here
```

### 4. Running Notebooks

```bash
pip install jupyterlab
jupyter lab
```

JupyterLab will open in your browser. Navigate to the `notebooks/` folder inside any course directory to find the `.ipynb` files. Alternatively, open notebooks directly in VS Code using the Jupyter extension.

### 5. Course Structure

| Course | Week | Python File | Notebook | Key Topic |
|--------|------|-------------|----------|-----------|
| Course 1 | Week 1 | `course1/code/week1_llm_foundations.py` | `course1/notebooks/week1_llm_foundations.ipynb` | LLM Foundations |
| Course 1 | Week 2 | `course1/code/week2_prompt_engineering.py` | `course1/notebooks/week2_prompt_engineering.ipynb` | Prompt Engineering |
| Course 1 | Week 3 | `course1/code/week3_ai_apis_scale.py` | `course1/notebooks/week3_ai_apis_scale.ipynb` | AI APIs at Scale |
| Course 1 | Week 4 | `course1/code/week4_embeddings_vectors.py` | `course1/notebooks/week4_embeddings_vectors.ipynb` | Embeddings & Vectors |
| Course 1 | Week 5 | `course1/code/week5_rag_pipeline.py` | `course1/notebooks/week5_rag_pipeline.ipynb` | RAG Pipeline |
| Course 1 | Week 6 | `course1/code/week6_fine_tuning.py` | `course1/notebooks/week6_fine_tuning.ipynb` | Fine-Tuning |
| Course 1 | Week 7 | `course1/code/week7_evaluation.py` | `course1/notebooks/week7_evaluation.ipynb` | Evaluation |
| Course 1 | Week 8 | `course1/code/week8_study_companion.py` | `course1/notebooks/week8_study_companion.ipynb` | Capstone: Study Companion |
| Course 2 | Week 1 | `course2/code/week1_tool_use.py` | `course2/notebooks/week1_tool_use.ipynb` | Tool Use |
| Course 2 | Week 2 | `course2/code/week2_mcp_server.py` | `course2/notebooks/week2_mcp_server.ipynb` | MCP Servers |
| Course 2 | Week 3 | `course2/code/week3_agent_foundations.py` | `course2/notebooks/week3_agent_foundations.ipynb` | Agent Foundations |
| Course 2 | Week 4 | `course2/code/week4_planning.py` | `course2/notebooks/week4_planning.ipynb` | Planning |
| Course 2 | Week 5 | `course2/code/week5_multi_agent.py` | `course2/notebooks/week5_multi_agent.ipynb` | Multi-Agent Systems |
| Course 2 | Week 6 | `course2/code/week6_agent_safety.py` | `course2/notebooks/week6_agent_safety.ipynb` | Agent Safety |
| Course 2 | Week 7 | `course2/code/week7_langgraph_frameworks.py` | `course2/notebooks/week7_langgraph_frameworks.ipynb` | Frameworks (LangGraph) |
| Course 2 | Week 8 | `course2/code/week8_research_agent.py` | `course2/notebooks/week8_research_agent.ipynb` | Capstone: Research Agent |
| Course 3 | Week 1 | `course3/code/week1_eval_at_scale.py` | `course3/notebooks/week1_eval_at_scale.ipynb` | Eval at Scale |
| Course 3 | Week 2 | `course3/code/week2_observability.py` | `course3/notebooks/week2_observability.ipynb` | Observability |
| Course 3 | Week 3 | `course3/code/week3_structured_generation.py` | `course3/notebooks/week3_structured_generation.ipynb` | Structured Generation |
| Course 3 | Week 4 | `course3/code/week4_safety_engineering.py` | `course3/notebooks/week4_safety_engineering.ipynb` | Safety Engineering |
| Course 3 | Week 5 | `course3/code/week5_deployment.py` | `course3/notebooks/week5_deployment.ipynb` | Deployment |
| Course 3 | Week 6 | `course3/code/week6_multimodal.py` | `course3/notebooks/week6_multimodal.ipynb` | Multimodal |
| Course 3 | Week 7 | `course3/code/week7_career_tools.py` | `course3/notebooks/week7_career_tools.ipynb` | Career Tools |
| Course 3 | Week 8 | `course3/code/week8_production_capstone.py` | `course3/notebooks/week8_production_capstone.ipynb` | Capstone: Production System |

### 6. Model Reference

| Model Name | Use Case | Cost Tier | Notes |
|------------|----------|-----------|-------|
| `mistral-large-latest` | Complex reasoning, instruction following, multi-step tasks | High | Best quality; use for capstone projects and hard tasks |
| `mistral-small-latest` | General-purpose chat, simpler completions, prototyping | Low | Good balance of speed and quality for most exercises |
| `codestral-latest` | Code generation, completion, debugging, code explanation | Medium | Specialized for programming tasks; supports fill-in-the-middle |
| `mistral-embed` | Text embeddings for semantic search, RAG, clustering | Low | Used in Week 4 and Week 5 (embeddings and RAG pipeline) |
| `pixtral-12b-2409` | Multimodal tasks — images + text, visual question answering | Medium | Used in Course 3 Week 6 (multimodal); accepts image inputs |

### 7. Troubleshooting

**API key not found (`AuthenticationError` or `KeyError: 'MISTRAL_API_KEY'`)**

- Confirm your `.env` file exists in the course directory and contains `MISTRAL_API_KEY=...` with no extra spaces or quotes.
- Make sure you are loading the `.env` file in your script: `from dotenv import load_dotenv; load_dotenv()`.
- In a notebook, call `load_dotenv()` in the first cell before any API calls.
- On Windows, confirm the `.env` file was not saved as `.env.txt` — enable file extensions in File Explorer to check.

**Rate limit errors (`429 Too Many Requests`)**

- Free-tier accounts have lower rate limits. Add a short `time.sleep(1)` between requests when running batch operations.
- Check your usage and quota at [console.mistral.ai](https://console.mistral.ai) under **Usage**.
- If doing embedding or evaluation at scale (Course 3 Week 1), batch your requests rather than sending them one at a time.

**Import errors (`ModuleNotFoundError`)**

- Ensure your virtual environment is activated before running scripts or launching JupyterLab.
- Run `pip install -r <course>/requirements.txt` for the specific course you are working in.
- In VS Code, select the correct Python interpreter (bottom-left status bar) pointing to your `venv`.

**Notebook kernel issues (kernel not starting or dying)**

- Confirm JupyterLab is installed in the same virtual environment as your course dependencies: `pip install jupyterlab`.
- Restart the kernel from the **Kernel** menu if a cell hangs.
- If the kernel repeatedly crashes on import, check for version conflicts: `pip list` and compare against `requirements.txt`.
- On Windows, if `jupyter lab` is not found after install, try running it as `python -m jupyter lab`.

**MCP server issues (Course 2 Week 2)**

- Ensure Node.js 20+ is installed: `node --version`.
- Install MCP server dependencies from within the `course2/code/` directory before starting the server.
- Check that the port used by the MCP server is not already in use.
