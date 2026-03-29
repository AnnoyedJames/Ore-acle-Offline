"""
Embedding protocol definitions for swappable embedding models.

Concrete implementations (SentenceTransformer-based, multimodal, API-based)
must satisfy :class:`EmbedderProtocol` to be usable in the pipeline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Contract for any embedding implementation.

    At minimum an embedder must be able to:
    1. Embed a batch of passage texts    → ``embed_passages``
    2. Embed a single query string       → ``embed_query``
    3. Report its output dimensionality  → ``dimension``
    """

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed a list of passage texts.

        Parameters
        ----------
        texts:
            Pre-formatted passage strings (prefixes already applied).

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), dimension)``, L2-normalised float32.
        """
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string for retrieval.

        Returns
        -------
        np.ndarray
            Shape ``(dimension,)``, L2-normalised float32.
        """
        ...
