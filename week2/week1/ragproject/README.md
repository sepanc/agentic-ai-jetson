# RAG Agent (Week 1)

A Retrieval-Augmented Generation (RAG) agent that loads PDFs, builds a vector store with ChromaDB, and answers questions using an Ollama LLM with source attribution. Suited for local development (e.g. M4 Mac) and deployment to edge devices (e.g. NVIDIA Jetson).

## Features

- **PDF document processing** — Load and chunk PDFs from a configurable directory
- **Vector store** — ChromaDB with HuggingFace `all-MiniLM-L6-v2` embeddings
- **Ollama LLM** — Local or remote Ollama API for generation
- **Source attribution** — Returns source documents and page numbers with each answer
- **Interactive CLI** — Query documents from the command line
- **Logging** — File and console logging for debugging

## Requirements

- Python 3.12+
- Ollama server (local or remote)
- Sufficient disk space for the vector store and embedding model

## Installation

1. **From the AILearn repo root, go to the project:**
   ```bash
   cd week1/ragproject
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment** — Create a `.env` file in this directory:
   ```env
   DOCUMENTS_DIR=./documents
   PERSIST_DIR=./data/chroma_db
   MODEL_NAME=llama3.2:3b
   OLLAMA_BASE_URL=http://localhost:11434
   DEVICE=mps   # Use 'cuda' on Jetson, 'mps' on M4 Mac, 'cpu' otherwise
   ```

5. **Add PDFs** — Place PDF files in the `documents/` directory.

6. **Ollama** — Install Ollama, run it, and pull the model:  
   `ollama pull llama3.2:3b`

## Project Structure

```
week1/ragproject/
├── documents/           # PDF files (add your own)
├── data/chroma_db/      # Vector store (created on first run)
├── logs/                # rag_agent.log
├── rag_agent.py         # Main RAG agent
├── test_rag.py          # Tests
├── requirements.txt
├── .env                 # You create this
└── README.md
```

## Usage

### Interactive mode

```bash
python rag_agent.py
```

The agent loads PDFs from `documents/`, builds or loads the vector store, then enters an interactive loop. Type questions and get answers with cited sources.

### Programmatic usage

```python
from rag_agent import RAGAgent

agent = RAGAgent()
agent.load_documents()
agent.create_vectorstore(agent.load_documents())
agent.setup_qa_chain()

result = agent.query("What is the main topic of these documents?")
print(result['answer'])
print(result['sources'])
```

### Tests

```bash
python test_rag.py
```

## Configuration

| Variable         | Default                 | Description                    |
|------------------|-------------------------|--------------------------------|
| `DOCUMENTS_DIR`  | `./documents`           | Directory containing PDFs      |
| `PERSIST_DIR`    | `./data/chroma_db`      | ChromaDB persistence path      |
| `MODEL_NAME`     | `llama3.2:3b`           | Ollama model name              |
| `OLLAMA_BASE_URL`| `http://localhost:11434`| Ollama API URL                 |
| `DEVICE`         | `mps`                   | `mps`, `cuda`, or `cpu`        |

## Acknowledgments

- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Ollama](https://ollama.ai/)
- [HuggingFace](https://huggingface.co/) embeddings
