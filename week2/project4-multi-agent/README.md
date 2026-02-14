# Project 4: Multi-Agent System (Week 2)

**Multi-agent** workflow: Research → Analysis → Report. Each agent has a dedicated role and uses shared tools; state is passed between agents to produce a final report.

## Features

- **Research agent** — Gathers information via search_document, database_query, api_call, web_search
- **Analysis agent** — Analyzes research data (calculate, summarize, compare, extract)
- **Report agent** — Formats findings (summary, detailed, bullet_points)
- **Shared state** — `MultiAgentState` carries messages and results between agents
- **Ollama** — All agents use local Ollama (e.g. `llama3.2:3b`) at `http://localhost:11434`

## Requirements

- Python 3.12+
- Ollama running at `http://localhost:11434` with a model such as `llama3.2:3b`

## Installation

```bash
cd week2/project4-multi-agent
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the multi-agent pipeline (entry point may be a script that invokes the research → analysis → report flow). See `agents.py` and `tools_extended.py` for the workflow.

## Project structure

- `agents.py` — Research, Analysis, and Report agents and orchestration
- `multi_agent_state.py` — Shared state and message handling
- `tools_extended.py` — Shared tools used by the agents
- `requirements.txt` — Dependencies

## Configuration

- **OLLAMA_BASE_URL** — Default `http://localhost:11434`. Override via environment or `.env` for remote Ollama.
