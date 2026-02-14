# Project 1: Structured Output (Week 2)

Vision-style detection agent that returns **validated, structured JSON** from an LLM using Pydantic. Demonstrates schema enforcement, retry-on-validation-failure, and clean extraction of JSON from model output.

## Features

- **Pydantic schemas** — `BoundingBox`, `Detection`, `VisionResponse` with strict validation (confidence 0–1, action one of track/ignore/alert)
- **Structured vision agent** — Sends a scene description to Ollama and parses/validates the response as JSON
- **Retry logic** — On validation failure, sends error feedback to the LLM and retries (configurable `max_retries`)
- **JSON extraction** — Handles markdown code blocks and surrounding text in LLM output

## Requirements

- Python 3.12+
- Ollama running locally with a suitable model (e.g. `llama3.2:3b`)

## Installation

1. **Enter the project directory:**
   ```bash
   cd week2/project1-structured-output
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -e .
   ```
   Or with uv: `uv sync`

3. **Ensure Ollama is running** at `http://localhost:11434` and has the model:  
   `ollama pull llama3.2:3b`

## Project Structure

```
week2/project1-structured-output/
├── structured_vision.py   # Pydantic models + StructuredVisionAgent
├── main.py                # Optional entry point
├── pyproject.toml
├── uv.lock
└── README.md
```

## Usage

Run the built-in demo (scene description → structured detections):

```bash
python structured_vision.py
```

Example: the agent gets a scene like *"Camera sees a person standing near a car in a parking lot"* and returns a `VisionResponse` with `detections` (object, confidence, bounding_box, action), `timestamp`, and `frame_id`.

### Use the agent in code

```python
from structured_vision import StructuredVisionAgent, VisionResponse

agent = StructuredVisionAgent(ollama_base_url="http://localhost:11434")
result: VisionResponse | None = agent.detect_objects(
    "A dog and a person in a garden",
    max_retries=3
)
if result:
    for d in result.detections:
        print(f"{d.object}: {d.confidence:.2f} -> {d.action}")
```

## Configuration

- **Ollama URL** — Pass `ollama_base_url` into `StructuredVisionAgent()` (default: `http://localhost:11434`).
- **Model** — Set inside `StructuredVisionAgent` (default: `llama3.2:3b`).
- **Retries** — `detect_objects(..., max_retries=3)`.

## Dependencies

- `langchain-core` — LLM invocation
- `langchain-ollama` — Ollama chat model
- `pydantic` — Structured output and validation
