"""
RAG Agent - M4 Development Version
Optimized for Python 3.12 + latest LangChain
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load environment
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rag_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RAGAgent:
    """
    RAG Agent for NVIDIA NCP-AAI Certification
    Development: M4 Mac → Jetson Ollama API
    Production: Deploy to Jetson (standalone)
    """
    
    def __init__(
        self,
        documents_dir: str = None,
        persist_dir: str = None,
        model_name: str = None,
        ollama_base_url: str = None,
        device: str = None
    ):
        # Configuration from .env or defaults
        self.documents_dir = Path(documents_dir or os.getenv('DOCUMENTS_DIR', './documents'))
        self.persist_dir = Path(persist_dir or os.getenv('PERSIST_DIR', './data/chroma_db'))
        self.model_name = model_name or os.getenv('MODEL_NAME', 'llama3.2:3b')
        self.ollama_base_url = ollama_base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.device = device or os.getenv('DEVICE', 'mps')
        
        logger.info("=" * 60)
        logger.info("RAG Agent Configuration:")
        logger.info(f"  LLM Endpoint: {self.ollama_base_url}")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Documents: {self.documents_dir}")
        logger.info("=" * 60)
        
        # Initialize components
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        
        self._setup_components()
        
    def _setup_components(self):
        """Initialize embeddings and LLM"""
        
        # Embeddings (local on M4)
        logger.info("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"✅ Embeddings loaded on {self.device}")
        
        # LLM (Jetson via API)
        logger.info(f"Connecting to Ollama at {self.ollama_base_url}...")
        try:
            self.llm = OllamaLLM(
                model=self.model_name,
                base_url=self.ollama_base_url,
                temperature=0.7
            )
            
            # Test connection
            test = self.llm.invoke("Hi")
            logger.info(f"✅ LLM connected: {test[:30]}...")
            
        except Exception as e:
            logger.error(f"❌ LLM connection failed: {e}")
            raise
    
    def load_documents(self) -> List[Any]:
        """Load PDF documents"""
        logger.info(f"Loading PDFs from {self.documents_dir}...")
        
        documents = []
        pdf_files = list(self.documents_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"⚠️  No PDFs found in {self.documents_dir}")
            return documents
        
        for pdf_path in pdf_files:
            logger.info(f"  📄 {pdf_path.name}...")
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load() 
                documents.extend(docs)
                logger.info(f"     ✅ {len(docs)} pages")
            except Exception as e:
                logger.error(f"     ❌ Error: {e}")
        
        logger.info(f"✅ Total: {len(documents)} pages loaded")
        return documents
    
    def create_vectorstore(self, documents: List[Any], force_recreate: bool = False):
        """Create or load vector store"""
        
        if self.persist_dir.exists() and not force_recreate:
            logger.info("📂 Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings
            )
            logger.info("✅ Loaded from disk")
        else:
            logger.info("🔨 Creating new vector store...")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            texts = text_splitter.split_documents(documents)
            logger.info(f"   Created {len(texts)} chunks")
            
            # Create embeddings
            logger.info(f"   Embedding on {self.device} (may take a few minutes)...")
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=str(self.persist_dir)
            )
            logger.info("✅ Vector store created")
        
        # Setup retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        logger.info("✅ Retriever ready (top-3 similarity)")
    
    def setup_qa_chain(self):
        """Setup QA chain"""
        logger.info("🔗 Setting up QA chain...")
        
        prompt_template = """Use the following context to answer the question.
If you don't know the answer, say so - don't make things up.

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("✅ QA chain ready")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        logger.info(f"💬 Query: {question}")
        
        if self.qa_chain is None:
            raise ValueError("QA chain not setup. Call setup_qa_chain() first.")
        
        try:
            result = self.qa_chain.invoke({"query": question})
            
            response = {
                "question": question,
                "answer": result["result"],
                "sources": [
                    {
                        "file": Path(doc.metadata.get("source", "Unknown")).name,
                        "page": doc.metadata.get("page", "?")
                    }
                    for doc in result["source_documents"]
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("✅ Query processed")
            return response
            
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "sources": [],
                "timestamp": datetime.now().isoformat()
            }


def main():
    """Interactive RAG"""
    
    print("\n" + "=" * 60)
    print("🤖 RAG Agent - NVIDIA NCP-AAI Demo")
    print("=" * 60)
    
    # Initialize
    agent = RAGAgent()
    
    # Load documents
    print("\n📚 Loading documents...")
    docs = agent.load_documents()
    
    if not docs:
        print(f"\n⚠️  No documents found!")
        print(f"   Add PDFs to: {agent.documents_dir.absolute()}")
        return
    
    # Create vector store
    print("\n🔨 Creating vector store...")
    agent.create_vectorstore(docs)
    
    # Setup QA
    print("\n🔗 Setting up QA chain...")
    agent.setup_qa_chain()
    
    print("\n" + "=" * 60)
    print("✅ Ready! Type 'quit' to exit")
    print("=" * 60)
    
    # Interactive loop
    while True:
        print("\n" + "-" * 60)
        question = input("💬 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not question:
            continue
        
        print("\n🤖 Thinking...")
        result = agent.query(question)
        
        print(f"\n✨ Answer:\n{result['answer']}")
        
        if result['sources']:
            print(f"\n📚 Sources:")
            for i, src in enumerate(result['sources'], 1):
                print(f"  [{i}] {src['file']} (Page {src['page']})")


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    main()