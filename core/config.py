import os
from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """
    Configuration settings for the application, loaded from a .env file.
    """
    # API Keys
    GOOGLE_API_KEY: str

    # LangChain Settings
    # LangChain Settings
    # LangChain Settings
    MODEL_NAME: str = "gemini-1.5-flash-latest" # This will be overridden in qa_engine, but keep it simple here.
    EMBED_MODEL_NAME: str = "models/text-embedding-004" # <-- THIS IS THE KEY FIX
        
    # Vector Store Settings
    DB_PERSIST_PATH: str = os.path.join(BASE_DIR, "vector_store")
    
    # Data Ingestion Settings
    DATA_PATH: str = os.path.join(BASE_DIR, "data/qms_documents")
    
    # QA Chain Settings
    SEARCH_K: int = 4

    class Config:
        env_file = os.path.join(BASE_DIR, ".env")
        env_file_encoding = "utf-8"

def get_settings() -> Settings:
    """Returns the settings instance."""
    return Settings()