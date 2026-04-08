"""
Centralized configuration for the Ore-acle RAG pipeline.

Loads secrets from .env, provides typed settings for all modules.
Usage:
    from config.settings import settings
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding Model Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingModelInfo:
    """Describes an embedding model used in the ablation study."""
    model_id: str
    dimension: int
    backend: str  # "local" (sentence-transformers) or "api" (OpenRouter)
    task_prefix: str = ""
    query_prefix: str = ""


EMBEDDING_MODELS: dict[str, EmbeddingModelInfo] = {
    "BAAI/bge-m3": EmbeddingModelInfo(
        model_id="BAAI/bge-m3",
        dimension=1024,
        backend="local",
    ),
    "nomic-ai/nomic-embed-text-v1.5": EmbeddingModelInfo(
        model_id="nomic-ai/nomic-embed-text-v1.5",
        dimension=768,
        backend="local",
        task_prefix="search_document: ",
        query_prefix="search_query: ",
    ),
    "intfloat/multilingual-e5-large": EmbeddingModelInfo(
        model_id="intfloat/multilingual-e5-large",
        dimension=1024,
        backend="local",
        task_prefix="passage: ",
        query_prefix="query: ",
    ),
    "google/gemini-embedding-001": EmbeddingModelInfo(
        model_id="google/gemini-embedding-001",
        dimension=3072,
        backend="api",
    ),
    # BGE-M3 via OpenRouter API (same model as BAAI/bge-m3 local, different backend)
    "baai/bge-m3": EmbeddingModelInfo(
        model_id="baai/bge-m3",
        dimension=1024,
        backend="api",
    ),
}

DEFAULT_EMBEDDING_MODEL = "baai/bge-m3"

# ---------------------------------------------------------------------------
# LLM Model Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMModelInfo:
    """Describes an LLM used in the generation evaluation."""
    model_id: str            # OpenRouter ID or Ollama tag
    backend: str             # "ollama" or "openrouter"
    label: str               # Short human-readable label for reports
    param_billions: float    # Approximate parameter count (for plots)


LLM_MODELS: dict[str, LLMModelInfo] = {
    "qwen3-0.6b": LLMModelInfo(
        model_id="qwen3:0.6b",
        backend="ollama",
        label="Qwen3 0.6B",
        param_billions=0.6,
    ),
    "qwen3-1.7b": LLMModelInfo(
        model_id="qwen3:1.7b",
        backend="ollama",
        label="Qwen3 1.7B",
        param_billions=1.7,
    ),
    "qwen3-4b": LLMModelInfo(
        model_id="qwen3:4b",
        backend="ollama",
        label="Qwen3 4B",
        param_billions=4.0,
    ),
    "qwen3-32b": LLMModelInfo(
        model_id="qwen/qwen3-32b",
        backend="openrouter",
        label="Qwen3 32B",
        param_billions=32.0,
    ),
    "gemini-flash-lite": LLMModelInfo(
        model_id="google/gemini-3.1-flash-lite-preview",
        backend="openrouter",
        label="Gemini 3.1 Flash Lite",
        param_billions=0,  # proprietary, size unknown
    ),
}

DEFAULT_LLM = "qwen3-4b"


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # --- OpenRouter (evals & generation gateway) ---
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")

    # --- Embedding ---
    embedding_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        description="Model ID — must be a key in EMBEDDING_MODELS",
    )
    embedding_dim: int = Field(default=1024, description="Embedding vector dimensions")
    embedding_batch_size: int = Field(
        default=128, description="Batch size for embedding generation"
    )
    embedding_task_prefix: str = Field(
        default="",
        description="Prefix prepended to passage texts before embedding",
    )
    embedding_query_prefix: str = Field(
        default="",
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

    def get_model_embeddings_dir(self, model_name: str = None) -> Path:
        """Get the specific directory where a model's embeddings are stored."""
        model = model_name or self.embedding_model
        safe_name = model.replace("/", "_").replace("\\", "_")
        return self.embeddings_dir / safe_name

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

# Singleton instance  import this everywhere
settings = Settings()

