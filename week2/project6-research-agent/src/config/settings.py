from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    
    # APIs
    serper_api_key: str
    
    # Paths
    project_root: Path = Path(__file__).parent.parent.parent
    chroma_persist_dir: Path = project_root / "data" / "chroma_db"
    knowledge_base_dir: Path = project_root / "data" / "knowledge_base"
    metrics_dir: Path = project_root / "data" / "deployment_metrics"
    
    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()