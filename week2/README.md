# AILearn

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/Ollama-LLM%20backend-000000?logo=ollama)](https://ollama.ai/)
[![LangChain](https://img.shields.io/badge/LangChain-chains%20%26%20agents-1C3C3C?logo=langchain)](https://www.langchain.com/)

Course and project materials for AI/ML learning: RAG, structured output, ReAct agents, and multi-source research.

## Repository structure

```
ailearn/
├── README.md                 # This file
├── requirements.txt          # Shared / reference dependencies
├── .gitignore
├── week1/
│   └── ragproject/           # RAG agent (PDFs + ChromaDB + Ollama)
│       └── README.md
└── week2/
    ├── project1-structured-output/   # Structured vision-style output (Pydantic + Ollama)
    │   └── README.md
    ├── project2-react-agent/         # ReAct agent (tools + loop detection)
    │   └── README.md
    ├── project3-multi-tool/          # Multi-tool ReAct
    ├── project4-multi-agent/          # Multi-agent
    ├── project5-integration/          # Integration
    └── project6-research-agent/       # Multi-source research (LangGraph)
        └── README.md
```

## Projects with READMEs

| Project | Path | Description |
|--------|------|-------------|
| RAG Agent | [week1/ragproject](week1/ragproject/README.md) | PDF → vector store → QA with Ollama and source attribution |
| Structured Output | [week2/project1-structured-output](week2/project1-structured-output/README.md) | Pydantic-validated JSON from LLM with retry logic |
| ReAct Agent | [week2/project2-react-agent](week2/project2-react-agent/README.md) | Thought–Action–Observation agent with calculator, search, datetime, web_search |
| Research Agent | [week2/project6-research-agent](week2/project6-research-agent/README.md) | LangGraph research pipeline: vector + arXiv + web + file search → report |

## Setup

- **Per-project install is recommended.** Each project may use its own venv and `requirements.txt` or `pyproject.toml`.
- For a single environment at repo root you can use:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- See each project’s README for dependencies, `.env`, and run instructions.

## Requirements

- Python 3.12+
- [Ollama](https://ollama.ai/) (for LLM-backed projects)
- Project-specific: Serper API key (project6), PDFs (week1, project6), etc. — see project READMEs.
