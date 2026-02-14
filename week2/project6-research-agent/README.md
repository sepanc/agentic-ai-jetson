# Project 6: Multi-Source Research Agent (Week 2)

A **research agent** that takes a question, classifies it, and runs a **LangGraph** workflow to query multiple sources (vector store, arXiv, web search, local files), then synthesizes a structured report with findings and recommendations.

## Features

- **Query analysis** — LLM classifies the query (academic / general / hybrid) and decides which sources to use
- **Multi-source retrieval** — Vector search (ChromaDB over local PDFs), arXiv API, web search (Serper), and file search
- **LangGraph orchestration** — Pipeline: analyze → plan retrieval → execute retrieval → synthesize → format report
- **Structured report** — Pydantic `ResearchReport`: category, confidence, sources, key findings, recommendations
- **CLI** — Run from the command line and optionally save the report to a Markdown file
- **Docker / Jetson** — Dockerfile and scripts for containerized and edge deployment

## Requirements

- Python 3.12+
- Ollama running (e.g. `llama3.2:3b`)
- **Serper API key** for web search (get one at [serper.dev](https://serper.dev))
- Optional: PDFs in `data/knowledge_base/` for vector search

## Installation

1. **Enter the project directory:**
   ```bash
   cd week2/project6-research-agent
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment** — Create a `.env` file:
   ```env
   SERPER_API_KEY=your_serper_api_key_here
   ```
   Optional overrides (defaults shown):
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2:3b
   ```

4. **Ollama** — Ensure Ollama is running and the model is pulled:  
   `ollama pull llama3.2:3b`

5. **Knowledge base (optional)** — Place PDFs in `data/knowledge_base/` for vector search. The project includes sample PDFs (e.g. gpt3.pdf, bert.pdf, transformer.pdf) for testing.

## Project Structure

```
week2/project6-research-agent/
├── run_research.py           # CLI entry point
├── src/
│   ├── agents/
│   │   └── research_agent.py # LangGraph workflow + ResearchAgent
│   ├── config/
│   │   └── settings.py       # Pydantic settings (.env)
│   ├── models/
│   │   └── schemas.py        # ResearchState, ResearchReport, etc.
│   └── tools/
│       ├── vector_search.py  # ChromaDB over knowledge_base PDFs
│       ├── arxiv_search.py   # arXiv API
│       ├── web_search.py    # Serper web search
│       ├── file_search.py   # Local file search
│       └── serper_search.py # Serper client
├── data/
│   ├── knowledge_base/       # PDFs for vector search
│   ├── chroma_db/           # Vector store (created on first run)
│   └── deployment_metrics/  # Optional constraints/costs data
├── tests/
│   ├── test_tools.py
│   └── test_schemas.py
├── requirements.txt
├── dockerfile
├── dockercompose.yml
├── run-research-docker.sh
├── deploy-to-jetson.sh
└── README.md
```

## Usage

### Basic run (report printed and saved to file)

```bash
python run_research.py "What are the main differences between BERT and GPT-3?"
```

### Options

```bash
python run_research.py "Your question here" \
  --output my_report.md \
  --model llama3.2:3b \
  --verbose
```

- `--output` — Output Markdown file (default: `research_report.md`)
- `--model` — Ollama model name
- `--verbose` — More logging

### Use the agent in code

```python
from src.agents.research_agent import ResearchAgent

agent = ResearchAgent(model="llama3.2:3b")
report = agent.research("How to deploy LLMs on edge devices?")

print(report.to_markdown())
print(report.category, report.confidence, report.key_findings)
```

### Tests

```bash
pytest tests/
```

## Configuration (settings / .env)

| Variable           | Default                 | Description              |
|--------------------|-------------------------|--------------------------|
| `SERPER_API_KEY`   | (required)               | Serper API key for web   |
| `OLLAMA_BASE_URL`  | `http://localhost:11434`| Ollama API URL           |
| `OLLAMA_MODEL`     | `llama3.2:3b`           | Ollama model             |

Paths (in `src/config/settings.py`) point at `data/chroma_db`, `data/knowledge_base`, and `data/deployment_metrics` under the project root.

## Docker / Jetson

- **dockerfile** — Build a container that runs the research agent.
- **dockercompose.yml** — Compose file for running the service.
- **run-research-docker.sh** — Helper to run research in Docker.
- **deploy-to-jetson.sh** — Helper for deploying to NVIDIA Jetson.

## Dependencies

- LangChain / LangGraph — Orchestration and LLM
- ChromaDB, sentence-transformers — Vector store and embeddings
- Pydantic / pydantic-settings — Schemas and config
- arxiv, httpx, requests — APIs and web
- pypdf — PDF handling for knowledge base
