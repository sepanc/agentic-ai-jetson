# Project 3: Multi-Tool ReAct Agent (Week 2)

ReAct agent with **multiple tools** (calculator, search, datetime, web search, etc.) and optional model comparison. Builds on Project 2 with extended tool set and comparison runs.

## Features

- **Multi-tool ReAct loop** — Same Thought → Action → Observation pattern with more tools
- **Extended tools** — From Project 2 plus additional utilities as needed
- **Model comparison** — Optional side-by-side runs for different Ollama models
- **Ollama** — Uses local Ollama (e.g. `llama3.2:3b`) at `http://localhost:11434`

## Requirements

- Python 3.12+
- Ollama running at `http://localhost:11434` with a model such as `llama3.2:3b`

## Installation

```bash
cd week2/project3-multi-tool
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: set `OLLAMA_BASE_URL` in a `.env` file if Ollama is on another host.

## Usage

- Run the ReAct agent: see `main.py` or invoke the multi-tool agent module.
- Model comparison: `python model_comparison.py` (or as documented in the script).

## Project structure

- `react_agent_multi.py` — Multi-tool ReAct agent
- `tools_extended.py` — Tool definitions
- `model_comparison.py` — Optional model comparison
- `requirements.txt`, `pyproject.toml` — Dependencies

## Configuration

- **OLLAMA_BASE_URL** — Default `http://localhost:11434`. Override via environment or `.env` for remote Ollama.
