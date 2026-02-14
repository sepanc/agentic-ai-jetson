# Project 2: ReAct Agent (Week 2)

A **ReAct** (Reason + Act) agent that answers questions by looping: **Thought → Action → Observation** using a small set of tools. Includes loop detection, answer extraction, and both regex-based and structured (Pydantic) parsing.

## Features

- **ReAct loop** — Thought → Action → Observation until a final answer or max steps
- **Four tools** — `calculator`, `search_document`, `get_datetime`, `web_search` (simulated)
- **Two implementations** — Regex parsing and Pydantic-structured output
- **Loop detection** — Detects repeated states and stops or forces finish to avoid infinite loops
- **Ollama** — Uses local Ollama (e.g. `llama3.2:3b`) with JSON output

## Requirements

- Python 3.12+
- Ollama running at `http://localhost:11434` with a model that supports JSON (e.g. `llama3.2:3b`)

## Installation

1. **Enter the project directory:**
   ```bash
   cd week2/project2-react-agent
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -e .
   ```
   Or with uv: `uv sync`

3. **Run Ollama** and pull the model:  
   `ollama pull llama3.2:3b`

## Project Structure

```
week2/project2-react-agent/
├── react_agent.py      # ReActAgent (regex) + ReActAgentStructured (Pydantic)
├── tools.py            # Tool definitions: calculator, search_document, get_datetime, web_search
├── test_tools.py       # Unit tests for tools
├── main.py             # Simple entry point
├── pyproject.toml
├── uv.lock
└── README.md
```

## Usage

### Run the ReAct comparison (regex vs structured)

```bash
python react_agent.py
```

This runs sample queries through both agent implementations and prints steps and final answers.

### Test tools only

```bash
python test_tools.py
```

### Use in code

```python
from react_agent import ReActAgent, ReActAgentStructured

# Regex-based agent
agent = ReActAgent(ollama_base_url="http://localhost:11434", max_steps=10)
answer = agent.run("What is 25% of the number of days until February 21, 2026?")

# Structured (Pydantic) agent
agent_s = ReActAgentStructured(ollama_base_url="http://localhost:11434", max_steps=10)
answer_s = agent_s.run("What is 15 * 12?")
```

## Tools

| Tool               | Description                    |
|--------------------|--------------------------------|
| `calculator`       | Safe math (e.g. `"22 * 0.25"`) |
| `search_document`  | Search pre-loaded text         |
| `get_datetime`     | Current date/time              |
| `web_search`       | Simulated web search          |
| `finish`           | Return final answer and stop  |

## Configuration

- **Ollama** — `ollama_base_url` in constructor (default: `http://localhost:11434`).
- **Max steps** — `max_steps` in constructor (default: 5 in code; use 10 for harder queries).

## Dependencies

- `langchain-core`, `langchain-ollama` — LLM and invocation
- `pydantic` — Structured ReAct response model
