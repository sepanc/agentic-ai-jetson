# Project 5: Integration (Week 2)

**Integration** project combining multi-agent coordination (Project 4), multi-tool usage (Project 3), ReAct-style reasoning (Project 2), and structured output (Project 1). Runs a full pipeline with research, analysis, and report agents using an extended tool set.

## Features

- **Integrated agents** — Research, Analysis, and Report agents with full multi-tool support
- **Extended tools** — search_document, database_query, api_call, web_search, get_datetime, file_read, calculator
- **Structured outputs** — Pydantic-validated actions and report format
- **Ollama** — Uses local Ollama (e.g. `llama3.2:3b`) at `http://localhost:11434`

## Requirements

- Python 3.12+
- Ollama running at `http://localhost:11434` with a model such as `llama3.2:3b`

## Installation

```bash
cd week2/project5-integration
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

- Run the integrated pipeline: see `integrated_agents.py` and any main/CLI entry point.
- Tests: `python test_integrated.py` (or `pytest`) for example queries.

## Project structure

- `integrated_agents.py` — Enhanced research, analysis, and report agents
- `integrated_graph.py` — Graph/orchestration if present
- `multi_agent_state.py` — Shared state
- `tools_extended.py` — Tool definitions
- `test_integrated.py` — Example integration tests
- `requirements.txt` — Dependencies

## Configuration

- **OLLAMA_BASE_URL** — Default `http://localhost:11434`. Override via environment or `.env` for remote Ollama.
