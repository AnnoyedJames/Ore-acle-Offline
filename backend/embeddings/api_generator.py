"""API-based Embedding Generator — uses OpenRouter for embeddings.

Provides the same interface as EmbeddingGenerator but calls a remote API
(e.g. google/gemini-embedding-001) instead of a local sentence-transformers
model.  Saves .npy + chunk_ids.json in the same directory layout.

Usage:
    from backend.embeddings.api_generator import ApiEmbeddingGenerator
    gen = ApiEmbeddingGenerator(model_id="google/gemini-embedding-001")
    vecs = gen.embed_passages(["Hello world"])
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from backend.config.settings import EMBEDDING_MODELS, settings

logger = logging.getLogger(__name__)

# OpenRouter rate-limit defaults (requests per minute)
_DEFAULT_RPM = 60
_BATCH_SIZE = 128  # texts per API call (OpenRouter batches)


class ApiEmbeddingGenerator:
    """Generates embeddings via the OpenRouter embeddings API.

    Satisfies the same interface as ``EmbeddingGenerator``:
    ``embed_passages``, ``embed_query``, ``dimension``.
    """

    def __init__(
        self,
        model_id: str = "google/gemini-embedding-001",
        batch_size: int = _BATCH_SIZE,
        output_dir: Optional[Path] = None,
    ):
        info = EMBEDDING_MODELS.get(model_id)
        if info is None:
            raise ValueError(
                f"Unknown embedding model '{model_id}'. "
                f"Available: {list(EMBEDDING_MODELS)}"
            )
        if info.backend != "api":
            raise ValueError(
                f"Model '{model_id}' has backend='{info.backend}', not 'api'. "
                "Use EmbeddingGenerator for local models."
            )

        self.model_id = model_id
        self._info = info
        self.batch_size = batch_size
        self.output_dir = output_dir or settings.get_model_embeddings_dir(model_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._client: Optional[OpenAI] = None

    # ------------------------------------------------------------------
    # EmbedderProtocol interface
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        return self._info.dimension

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed a list of passage texts via the API.

        Applies task_prefix from the model registry if set.
        Returns an L2-normalised float32 array of shape ``(n, dim)``.
        """
        client = self._get_client()
        prefix = self._info.task_prefix

        all_embeddings: list[np.ndarray] = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(n_batches), desc=f"API embed ({self.model_id})"):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(texts))
            batch = [f"{prefix}{t}" for t in texts[start:end]]

            vecs = self._call_api(client, batch)
            all_embeddings.append(vecs)

        result = np.vstack(all_embeddings).astype(np.float32)
        # L2-normalise for cosine similarity
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        result /= norms
        return result

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string for retrieval."""
        client = self._get_client()
        prefix = self._info.query_prefix
        text = f"{prefix}{query}"
        vecs = self._call_api(client, [text])
        vec = vecs[0].astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    # ------------------------------------------------------------------
    # Bulk generate (mirrors EmbeddingGenerator.generate)
    # ------------------------------------------------------------------

    def generate(self, chunks_file: Optional[Path] = None, resume: bool = True) -> tuple[np.ndarray, list[str]]:
        """Generate embeddings for all chunks, with checkpoint/resume."""
        chunks_path = chunks_file or settings.chunks_file
        logger.info(f"Loading chunks from {chunks_path}")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        logger.info(f"Total chunks: {len(chunks)}")

        chunk_ids = [c["chunk_id"] for c in chunks]
        embeddings_path = self.output_dir / "embeddings.npy"
        ids_path = self.output_dir / "chunk_ids.json"

        start_idx = 0
        existing_embeddings = None

        if resume and embeddings_path.exists() and ids_path.exists():
            try:
                existing_embeddings = np.load(embeddings_path)
                with open(ids_path, "r") as f:
                    existing_ids = json.load(f)
                if existing_embeddings.shape[0] == len(existing_ids):
                    start_idx = len(existing_ids)
                    logger.info(f"Resuming: {start_idx}/{len(chunks)} done")
                else:
                    existing_embeddings = None
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                existing_embeddings = None

        if start_idx >= len(chunks):
            logger.info("All chunks already embedded!")
            return existing_embeddings if existing_embeddings is not None else np.load(embeddings_path), chunk_ids

        remaining = chunks[start_idx:]
        texts = self._prepare_texts(remaining)
        new_embeddings = self.embed_passages(texts)

        if existing_embeddings is not None:
            final = np.vstack([existing_embeddings, new_embeddings])
        else:
            final = new_embeddings

        np.save(embeddings_path, final)
        with open(ids_path, "w") as f:
            json.dump(chunk_ids, f)

        logger.info(f"Saved embeddings: {final.shape} to {embeddings_path}")
        return final, chunk_ids

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> OpenAI:
        if self._client is None:
            api_key = settings.openrouter_api_key
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY is not set")
            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        return self._client

    def _call_api(self, client: OpenAI, texts: list[str]) -> np.ndarray:
        """Call the OpenRouter embeddings endpoint with retry."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(model=self.model_id, input=texts)
                vecs = [item.embedding for item in resp.data]
                return np.array(vecs, dtype=np.float32)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"API error (attempt {attempt + 1}): {e}, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise

    @staticmethod
    def _prepare_texts(chunks: list[dict]) -> list[str]:
        """Combine page_title + section_heading + text (same as local generator)."""
        texts = []
        for chunk in chunks:
            parts = []
            title = chunk.get("page_title", "")
            section = chunk.get("section_heading", "")
            text = chunk.get("text", "")
            if title:
                parts.append(title)
            if section and section not in ("Introduction", "Infobox"):
                parts.append(f"- {section}")
            parts.append(text)
            texts.append("\n".join(parts))
        return texts
