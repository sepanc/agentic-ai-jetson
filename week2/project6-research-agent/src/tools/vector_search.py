"""
Vector search tool using ChromaDB for local knowledge base queries.
Builds on Week 1 RAG implementation.
"""

from typing import List
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging
import sys

if __name__ == "__main__":
    # Running directly - add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.models.schemas import VectorSource, SourceType
    from src.config.settings import settings
else:
    # Running as package import
    from ..models.schemas import VectorSource, SourceType
    from ..config.settings import settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorSearchTool:
    """Query local vector database (ChromaDB) for relevant documents"""
    
    def __init__(
        self,
        persist_dir: Path = None,
        embedding_model_name: str = None,
        collection_name: str = "research_docs"
    ):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.embedding_model_name = embedding_model_name or settings.embedding_model
        self.collection_name = collection_name
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Research documents knowledge base"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[dict],
        ids: List[str]
    ) -> None:
        """
        Add documents to vector database
        
        Args:
            documents: List of text content
            metadatas: List of metadata dicts (must include 'title', 'source')
            ids: List of unique document IDs
        """
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        ).tolist()
        
        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        logger.info(f"Added {len(documents)} documents to collection")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_relevance: float = 0.0
    ) -> List[VectorSource]:
        """
        Search vector database for relevant documents
        
        Args:
            query: Search query string
            top_k: Number of results to return
            min_relevance: Minimum relevance score (0-1, based on cosine similarity)
            
        Returns:
            List of VectorSource objects with results
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(
                query,
                convert_to_numpy=True
            ).tolist()
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to VectorSource objects
            sources = []
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                document = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Convert distance to similarity score (lower distance = higher similarity)
                # ChromaDB uses L2 distance, approximate conversion to 0-1 scale
                relevance_score = max(0.0, 1.0 - (distance / 2.0))
                
                # Skip if below minimum relevance
                if relevance_score < min_relevance:
                    continue
                
                source = VectorSource(
                    title=metadata.get('title', 'Untitled Document'),
                    content=document,
                    relevance_score=relevance_score,
                    url=metadata.get('url'),
                    metadata=metadata,
                    document_id=doc_id,
                    chunk_id=metadata.get('chunk_id'),
                    distance=distance
                )
                sources.append(source)
            
            logger.info(f"Vector search returned {len(sources)} results for query: {query[:50]}...")
            return sources
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the vector database"""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "embedding_dimension": len(self.embedding_model.encode("test")),
            "model": self.embedding_model_name
        }


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    # Test vector search tool
    tool = VectorSearchTool()
    
    # Add sample documents (if collection is empty)
    stats = tool.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    if stats['document_count'] == 0:
        print("\nAdding sample documents...")
        tool.add_documents(
            documents=[
                "NVIDIA Jetson is a compact AI computer for edge deployment. It offers GPU acceleration in a low-power form factor suitable for robotics and IoT applications.",
                "Transformer models like BERT use attention mechanisms to process sequential data. The self-attention layer allows the model to weigh the importance of different tokens.",
                "Edge AI brings computation closer to data sources, reducing latency and bandwidth requirements. This is critical for real-time applications and privacy-sensitive use cases."
            ],
            metadatas=[
                {"title": "Jetson Overview", "source": "nvidia.com", "topic": "hardware"},
                {"title": "Transformer Architecture", "source": "arxiv.org", "topic": "ml_theory"},
                {"title": "Edge AI Benefits", "source": "developer.nvidia.com", "topic": "deployment"}
            ],
            ids=["doc_1", "doc_2", "doc_3"]
        )
    
    # Test search
    print("\n" + "="*80)
    print("Testing vector search...")
    print("="*80)
    
    query = "What hardware is good for edge AI deployment?"
    results = tool.search(query, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.title} (relevance: {result.relevance_score:.2f})")
        print(f"   {result.content[:100]}...")
        print(f"   Distance: {result.distance:.4f}\n")