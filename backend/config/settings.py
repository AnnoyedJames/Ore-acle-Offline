"""
Centralized configuration for the Ore-acle RAG pipeline.

Loads secrets from .env, provides typed settings for all modules.
Usage:
    from config.settings import settings
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from pydantic import Field

# Ensure HuggingFace tools run fully offline, assuming cached models
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
    # BGE-M3 via OpenRouter API (same model as BAAI/bge-m3 local, different backend)
    "baai/bge-m3": EmbeddingModelInfo(
        model_id="baai/bge-m3",
        dimension=1024,
        backend="api",
    ),
}

DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"

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
    "gemma-4-e2b": LLMModelInfo(
        model_id="gemma4:e2b",
        backend="ollama",
        label="Gemma 4 e2B",
        param_billions=2.0,
    ),
    "gemma-4-e4b": LLMModelInfo(
        model_id="gemma4:e4b",
        backend="ollama",
        label="Gemma 4 e4B",
        param_billions=4.0,
    ),
    "gemma-4-31b": LLMModelInfo(
        model_id="google/gemma-4-31b-it",
        backend="openrouter",
        label="Gemma 4 31B",
        param_billions=31.0,
    ),
    "gemini-flash-lite": LLMModelInfo(
        model_id="google/gemini-3.1-flash-lite-preview",
        backend="openrouter",
        label="Gemini 3.1 Flash Lite",
        param_billions=0,  # proprietary, size unknown
    ),
}

DEFAULT_LLM = "gemini-flash-lite"


# ---------------------------------------------------------------------------
# Reranker Model Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RerankerModelInfo:
    """Describes a reranker used after first-stage retrieval."""
    model_id: str
    backend: str  # "local" (sentence-transformers CrossEncoder) or "zeroentropy"
    label: str
    quantization: Optional[str] = None  # "4bit", "8bit", or None


RERANKER_MODELS: dict[str, RerankerModelInfo] = {
    "bge-reranker-v2-m3": RerankerModelInfo(
        model_id="BAAI/bge-reranker-v2-m3",
        backend="local",
        label="BGE Reranker v2-m3",
    ),
    "qwen3-reranker-0.6b": RerankerModelInfo(
        model_id="Qwen/Qwen3-Reranker-0.6B",
        backend="local",
        label="Qwen3-Reranker-0.6B",
        quantization="4bit",
    ),
    "qwen3-reranker-4b": RerankerModelInfo(
        model_id="Qwen/Qwen3-Reranker-4B",
        backend="local",
        label="Qwen3-Reranker-4B",
        quantization="4bit",
    ),
    "zerank-2": RerankerModelInfo(
        model_id="zeroentropy/zerank-2",
        backend="local",
        label="ZeroEntropy zerank-2",
        quantization="4bit",
    ),
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # --- OpenRouter (evals & generation gateway) ---
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    zeroentropy_api_key: str = Field(default="", description="ZeroEntropy API key")

    # --- Embedding ---
    embedding_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        description="Model ID — must be a key in EMBEDDING_MODELS",
    )
    embedding_dim: int = Field(default=1024, description="Embedding vector dimensions")
    embedding_batch_size: int = Field(
        default=16, description="Batch size for embedding generation"
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
        default=15, description="Candidates from keyword search before RRF"
    )
    retrieval_rerank_candidates: int = Field(
        default=30,
        description="Number of fused candidates to pass into the reranker",
    )
    rrf_k: int = Field(
        default=20, description="RRF constant (higher = more weight to lower ranks)"
    )
    rrf_alpha: float = Field(
        default=0.8,
        description="Semantic weight in weighted RRF; keyword weight = 1 - rrf_alpha",
    )
    reranker_model: str = Field(
        default="",
        description="Optional reranker key from RERANKER_MODELS; blank disables reranking",
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

