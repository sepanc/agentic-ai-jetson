# RAG Agent for NVIDIA NCP-AAI

A Retrieval-Augmented Generation (RAG) agent optimized for NVIDIA NCP-AAI certification. This project demonstrates a production-ready RAG system that can be developed on M4 Mac and deployed to NVIDIA Jetson devices.

## 🚀 Features

- **PDF Document Processing**: Automatically loads and processes PDF documents from a directory
- **Vector Store**: Uses ChromaDB for efficient document embedding and retrieval
- **HuggingFace Embeddings**: Leverages `all-MiniLM-L6-v2` for fast, accurate embeddings
- **Ollama LLM Integration**: Connects to Ollama API (local or remote) for language model inference
- **Retrieval-Augmented Generation**: Combines document retrieval with LLM generation for accurate, context-aware answers
- **Source Attribution**: Provides source documents and page numbers for each answer
- **Flexible Deployment**: 
  - **Development**: Run on M4 Mac with local embeddings and remote Jetson LLM
  - **Production**: Deploy standalone on Jetson device
- **Interactive CLI**: User-friendly command-line interface for querying documents
- **Comprehensive Logging**: Detailed logs for debugging and monitoring

## 📋 Requirements

- Python 3.12+
- Ollama server (local or remote)
- NVIDIA Jetson device (for production deployment)
- Sufficient disk space for vector store and embeddings

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sepanc/rag-agent-nvidia-jetson.git
   cd rag-agent-nvidia-jetson
   ```

2. **Create a virtual environment:**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```env
   DOCUMENTS_DIR=./documents
   PERSIST_DIR=./data/chroma_db
   MODEL_NAME=llama3.2:3b
   OLLAMA_BASE_URL=http://localhost:11434
   DEVICE=mps  # Use 'cuda' for Jetson, 'mps' for M4 Mac, 'cpu' for CPU
   ```

5. **Prepare documents:**
   Place your PDF files in the `documents/` directory

6. **Set up Ollama:**
   - Install Ollama on your local machine or Jetson device
   - Pull the model: `ollama pull llama3.2:3b`
   - Ensure Ollama is running and accessible at the configured URL

## 📁 Project Structure

```
rag-agent-nvidia-jetson/
├── documents/              # PDF documents directory
│   ├── gpt3.pdf
│   └── transformer.pdf
├── data/                   # Vector store (auto-generated)
│   └── chroma_db/
├── logs/                   # Log files (auto-generated)
│   └── rag_agent.log
├── rag_agent.py           # Main RAG agent implementation
├── test_rag.py            # Test suite
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── .gitignore
└── README.md
```

## 🎯 Usage

### Interactive Mode

Run the interactive RAG agent:

```bash
python rag_agent.py
```

The agent will:
1. Load all PDFs from the `documents/` directory
2. Create or load the vector store
3. Set up the QA chain
4. Enter an interactive loop where you can ask questions

Example session:
```
💬 Your question: What is GPT-3?

✨ Answer:
GPT-3 (Generative Pre-trained Transformer 3) is a large language model...

📚 Sources:
  [1] gpt3.pdf (Page 1)
  [2] gpt3.pdf (Page 2)
```

### Programmatic Usage

```python
from rag_agent import RAGAgent

# Initialize agent
agent = RAGAgent()

# Load documents
docs = agent.load_documents()

# Create vector store
agent.create_vectorstore(docs)

# Setup QA chain
agent.setup_qa_chain()

# Query
result = agent.query("What is the main topic of these documents?")
print(result['answer'])
print(result['sources'])
```

### Testing

Run the test suite to verify installation:

```bash
python test_rag.py
```

This will test:
- Agent initialization
- Document loading
- Vector store creation
- QA chain setup
- Sample query execution

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCUMENTS_DIR` | `./documents` | Directory containing PDF files |
| `PERSIST_DIR` | `./data/chroma_db` | Vector store persistence directory |
| `MODEL_NAME` | `llama3.2:3b` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `DEVICE` | `mps` | Device for embeddings (`mps`, `cuda`, or `cpu`) |

### Customization

- **Chunk Size**: Modify `chunk_size` and `chunk_overlap` in `create_vectorstore()` method
- **Retrieval**: Adjust `k` parameter in `as_retriever()` for more/fewer source documents
- **Prompt Template**: Customize the prompt in `setup_qa_chain()` method
- **Embedding Model**: Change `model_name` in `_setup_components()` for different embeddings

## 🔄 Development Workflow

### M4 Mac Development

1. Run embeddings locally on M4 (using MPS)
2. Connect to Ollama API on Jetson device
3. Test and iterate quickly

### Jetson Deployment

1. Deploy code to Jetson
2. Set `DEVICE=cuda` in `.env`
3. Set `OLLAMA_BASE_URL=http://localhost:11434` for local Ollama
4. Run as standalone application

## 📝 Logging

Logs are written to:
- Console (stdout)
- `logs/rag_agent.log` file

Log levels include:
- INFO: General operations
- WARNING: Non-critical issues
- ERROR: Failures and exceptions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is part of the NVIDIA NCP-AAI certification program.

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Uses [ChromaDB](https://www.trychroma.com/) for vector storage
- Powered by [Ollama](https://ollama.ai/) for LLM inference
- Embeddings from [HuggingFace](https://huggingface.co/)

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This is a development version optimized for Python 3.12 and the latest LangChain libraries. Ensure all dependencies are up to date for best performance.
