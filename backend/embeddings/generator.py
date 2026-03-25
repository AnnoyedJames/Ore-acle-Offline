"""Embedding Generator - produces content embeddings for chunks.

Loads chunks from chunks.json, generates embeddings in batches using
sentence-transformers, and saves both the vectors (numpy .npy) and
a mapping file for database upload.

We use Multilingual-E5-Large (1024d) for high-quality retrieval.
Vectors are L2-normalized for cosine similarity.

Usage:
    python -m embeddings.generator
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    chunks_file: Path = field(
        default_factory=lambda: Path("data/processed/chunks.json")
    )
    output_dir: Path = field(
        default_factory=lambda: Path("data/processed/embeddings")
    )
    # Multilingual E5 Large — 1024d vectors supported by Pinecone Inference
    model_name: str = "intfloat/multilingual-e5-large"
    batch_size: int = 128  # Higher batch size for 4060
    device: str = ""  # Auto-detect: cuda if available, else cpu
    # Native dimension
    truncate_dim: int = 1024
    # E5 specific prefixes (passage for docs, query for search)
    task_prefix: str = "passage: "
    query_prefix: str = "query: "
    checkpoint_interval: int = 50


class EmbeddingGenerator:
    """Generates embeddings for chunks using Nomic v1.5.

    Saves:
      - embeddings.npy: float32 array of shape (num_chunks, 768)
      - chunk_ids.json: ordered list of chunk_ids matching embedding rows
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self.model is not None:
            return

        logger.info(f"Loading embedding model: {self.config.model_name}")
        from sentence_transformers import SentenceTransformer
        import torch

        # Auto-detect device if not explicitly set
        device = self.config.device
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.config.device = device

        # trust_remote_code=True is required for Nomic
        self.model = SentenceTransformer(
            self.config.model_name, 
            trust_remote_code=True,
            device=device,
        )

        logger.info(f"Model loaded on {device}")
        logger.info(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def _prepare_texts(self, chunks: list[dict]) -> list[str]:
        """
        Prepare texts for embedding.

        Combines page title + section heading + text for richer embeddings.
        Applies task prefix if configured (search_document: for Nomic).
        """
        texts = []
        for chunk in chunks:
            parts = []
            title = chunk.get("page_title", "")
            section = chunk.get("section_heading", "")
            text = chunk.get("text", "")

            if title:
                parts.append(title)
            if section and section != "Introduction" and section != "Infobox":
                parts.append(f"- {section}")
            
            # Note: Nomic prefers just text often, but with RAG context, title helps.
            # We keep the title/section.
            parts.append(text)

            combined = "\n".join(parts)
            if self.config.task_prefix:
                combined = f"{self.config.task_prefix}{combined}"
            texts.append(combined)

        return texts

    def generate(self, resume: bool = True) -> tuple[np.ndarray, list[str]]:
        """
        Generate embeddings for all chunks.
        
        Args:
            resume: If True, skip chunks that already have embeddings from
                    a previous checkpoint.
        
        Returns:
            (embeddings_array, chunk_ids_list)
        """
        self._load_model()

        # Load chunks
        logger.info(f"Loading chunks from {self.config.chunks_file}")
        with open(self.config.chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        logger.info(f"Total chunks: {len(chunks)}")

        chunk_ids = [c["chunk_id"] for c in chunks]

        # Check for existing checkpoint
        embeddings_path = self.config.output_dir / "embeddings.npy"
        ids_path = self.config.output_dir / "chunk_ids.json"
        checkpoint_path = self.config.output_dir / "checkpoint.npy"

        start_idx = 0
        existing_embeddings = None

        if resume and checkpoint_path.exists() and ids_path.exists():
            try:
                existing_embeddings = np.load(checkpoint_path)
                with open(ids_path, "r") as f:
                    existing_ids = json.load(f)
                start_idx = len(existing_ids)
                logger.info(f"Resuming from checkpoint: {start_idx}/{len(chunks)} done")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}, starting fresh")
                start_idx = 0
                existing_embeddings = None

        if start_idx >= len(chunks):
            logger.info("All chunks already embedded!")
            if existing_embeddings is not None:
                return existing_embeddings, chunk_ids
            else:
                embeddings = np.load(embeddings_path)
                return embeddings, chunk_ids

        # Prepare texts for remaining chunks
        remaining_chunks = chunks[start_idx:]
        texts = self._prepare_texts(remaining_chunks)

        logger.info(f"Embedding {len(texts)} chunks in batches of {self.config.batch_size}...")

        all_embeddings = []
        num_batches = (len(texts) + self.config.batch_size - 1) // self.config.batch_size

        for batch_idx in tqdm(range(num_batches), desc="Embedding batches"):
            batch_start = batch_idx * self.config.batch_size
            batch_end = min(batch_start + self.config.batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]

            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
                normalize_embeddings=True,  # L2-normalize for cosine similarity
            )

            all_embeddings.append(batch_embeddings)

            # Checkpoint
            if (batch_idx + 1) % self.config.checkpoint_interval == 0:
                partial = np.vstack(all_embeddings)
                if existing_embeddings is not None:
                    partial = np.vstack([existing_embeddings, partial])
                np.save(checkpoint_path, partial)
                processed_ids = chunk_ids[: start_idx + batch_end]
                with open(ids_path, "w") as f:
                    json.dump(processed_ids, f)
                logger.info(
                    f"Checkpoint saved: {start_idx + batch_end}/{len(chunks)}"
                )

        # Combine all
        new_embeddings = np.vstack(all_embeddings)
        if existing_embeddings is not None:
            final_embeddings = np.vstack([existing_embeddings, new_embeddings])
        else:
            final_embeddings = new_embeddings

        # Save final output
        np.save(embeddings_path, final_embeddings)
        with open(ids_path, "w") as f:
            json.dump(chunk_ids, f)

        # Clean up checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        logger.info(f"Saved embeddings: {final_embeddings.shape} to {embeddings_path}")
        logger.info(f"Saved chunk IDs: {len(chunk_ids)} to {ids_path}")

        return final_embeddings, chunk_ids

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string for retrieval.
        Uses query: prefix. Returns full 1024d L2-normalized vector.
        """
        self._load_model()
        prefixed = f"{self.config.query_prefix}{query}"
        embedding = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding[0]


if __name__ == "__main__":
    generator = EmbeddingGenerator()
    embeddings, ids = generator.generate()
    print(f"Generated {len(ids)} embeddings with shape {embeddings.shape}")
