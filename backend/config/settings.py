"""
Centralized configuration for the Ore-acle RAG pipeline.

Loads secrets from .env, provides typed settings for all modules.
Usage:
    from config.settings import settings
"""

import logging
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""    

    # --- OpenRouter (evals & generation gateway) ---
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")

    # --- Embedding ---
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-large",
        description="HuggingFace model ID for sentence-transformers",
    )
    embedding_dim: int = Field(default=1024, description="Embedding vector dimensions")
    embedding_batch_size: int = Field(
        default=128, description="Batch size for embedding generation"
    )
    embedding_task_prefix: str = Field(
        default="passage: ",
        description="Prefix prepended to passage texts before embedding",
    )
    embedding_query_prefix: str = Field(
        default="query: ",
        description="Prefix prepended to query texts before embedding",
    )
    embedding_device: str = Field(
        default="cuda",
        description="Device for embedding model: cpu, cuda, mps",
    )

    # --- Local Databases ---
    chroma_db_dir: Path = Field(
        default_factory=lambda: Path("data/chroma_db"),
        description="Directory for ChromaDB storage",
    )
    sqlite_db_path: Path = Field(
        default_factory=lambda: Path("data/sqlite_fts.db"),
        description="Path to SQLite FTS5 database",
    )

    # --- Chunking ---
    chunker_strategy: str = Field(
        default="section_aware",
        description="Chunking strategy: section_aware (default) or langchain",
    )
    chunk_max_tokens: int = Field(
        default=512, description="Maximum tokens per chunk"
    )
    chunk_overlap_tokens: int = Field(
        default=50, description="Token overlap between split chunks within a section"
    )

    # --- Retrieval ---
    retrieval_top_k: int = Field(
        default=10, description="Number of chunks to return from hybrid search"  
    )
    retrieval_semantic_candidates: int = Field(
        default=20, description="Candidates from semantic search before RRF"     
    )
    retrieval_keyword_candidates: int = Field(
        default=20, description="Candidates from keyword search before RRF"      
    )
    rrf_k: int = Field(
        default=60, description="RRF constant (higher = more weight to lower ranks)"
    )

    # --- Paths ---
    data_raw_dir: Path = Field(
        default_factory=lambda: Path("data/raw"),
        description="Raw data directory",
    )
    data_processed_dir: Path = Field(
        default_factory=lambda: Path("data/processed"),
        description="Processed data directory",
    )
    metadata_file: Path = Field(
        default_factory=lambda: Path("data/processed/metadata.json"),
        description="Text cleaner output",
    )
    chunks_file: Path = Field(
        default_factory=lambda: Path("data/processed/chunks.json"),
        description="Chunker output",
    )
    interlinks_file: Path = Field(
        default_factory=lambda: Path("data/processed/interlinks.json"),
        description="Interlink graph output",
    )
    embeddings_dir: Path = Field(
        default_factory=lambda: Path("data/processed/embeddings"),
        description="Directory for saved embedding arrays",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

# Singleton instance  import this everywhere
settings = Settings()

