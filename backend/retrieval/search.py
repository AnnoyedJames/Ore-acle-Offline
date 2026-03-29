"""
Local Hybrid Search — Reciprocal Rank Fusion over ChromaDB + SQLite FTS5.

Drop-in replacement for the cloud-based search module. Uses the same
SearchResult dataclass and RRF algorithm but queries local stores.

Usage:
    from backend.retrieval.local_search import LocalHybridSearch
    search = LocalHybridSearch()
    results = search.search("How do I find diamonds?")
"""

import json
import logging
from typing import Optional

import numpy as np

from backend.config.settings import settings
from backend.database.local_stores import ChromaStore, SQLiteStore
from backend.embeddings.generator import EmbeddingGenerator
from backend.retrieval.search import SearchResult

logger = logging.getLogger(__name__)


class HybridSearch:
    """
    Hybrid search combining semantic (ChromaDB) and keyword (SQLite FTS5)
    retrieval using Reciprocal Rank Fusion.

    RRF formula: score(d) = Σ 1/(k + rank_i(d))
    """

    def __init__(
        self,
        chroma: Optional[ChromaStore] = None,
        sqlite: Optional[SQLiteStore] = None,
        embedder: Optional[EmbeddingGenerator] = None,
    ):
        self.chroma = chroma or ChromaStore()
        self.sqlite = sqlite or SQLiteStore()
        self._embedder = embedder
        # Search parameters from settings
        self.semantic_candidates = settings.retrieval_semantic_candidates
        self.keyword_candidates = settings.retrieval_keyword_candidates
        self.top_k = settings.retrieval_top_k
        self.rrf_k = settings.rrf_k

    @property
    def embedder(self) -> EmbeddingGenerator:
        if self._embedder is None:
            self._embedder = EmbeddingGenerator()
        return self._embedder

    def _semantic_search(self, query: str) -> list[dict]:
        """Embed query and search ChromaDB."""
        query_vec = self.embedder.embed_query(query)
        results = self.chroma.query(query_vec, n_results=self.semantic_candidates)
        return results

    def _keyword_search(self, query: str) -> list[dict]:
        """Search SQLite FTS5 for matching chunks."""
        return self.sqlite.search(query, limit=self.keyword_candidates)

    def _rrf_merge(
        self,
        semantic_results: list[dict],
        keyword_results: list[dict],
        chunks_lookup: dict[str, dict],
    ) -> list[SearchResult]:
        """Merge results using Reciprocal Rank Fusion."""
        k = self.rrf_k
        scores: dict[str, float] = {}
        semantic_scores: dict[str, float] = {}
        keyword_scores: dict[str, float] = {}

        # Score semantic results
        for rank, result in enumerate(semantic_results):
            cid = result["id"]
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
            semantic_scores[cid] = 1.0 - result.get("distance", 0)

        # Score keyword results
        for rank, result in enumerate(keyword_results):
            cid = result["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
            keyword_scores[cid] = abs(result.get("rank", 0))

        # Sort by RRF score descending
        sorted_ids = sorted(scores.keys(), key=lambda x: -scores[x])
        top_ids = sorted_ids[: self.top_k]

        results = []
        for cid in top_ids:
            # Prefer semantic result metadata (has full fields),
            # fall back to chunks.json lookup
            sem = next(
                (r for r in semantic_results if r["id"] == cid), None
            )
            if sem and "text" in sem:
                c = sem
            elif cid in chunks_lookup:
                c = chunks_lookup[cid]
            else:
                logger.warning(f"Chunk {cid} not found in any source, skipping")
                continue

            # Parse JSON strings if needed (from ChromaDB metadata)
            images = c.get("images", [])
            if isinstance(images, str):
                try:
                    images = json.loads(images)
                except (json.JSONDecodeError, TypeError):
                    images = []

            infobox = c.get("infobox")
            if isinstance(infobox, str):
                try:
                    infobox = json.loads(infobox)
                except (json.JSONDecodeError, TypeError):
                    infobox = None

            results.append(SearchResult(
                chunk_id=cid,
                page_title=c.get("page_title", ""),
                page_url=c.get("page_url", ""),
                section_heading=c.get("section_heading", ""),
                section_level=c.get("section_level", 2),
                text=c.get("text", ""),
                token_count=c.get("token_count", 0),
                chunk_type=c.get("chunk_type", "section"),
                page_type=c.get("page_type", "other"),
                infobox=infobox,
                images=images,
                rrf_score=scores[cid],
                semantic_score=semantic_scores.get(cid),
                keyword_score=keyword_scores.get(cid),
            ))

        return results

    def search(
        self,
        query: str,
        filter_types: Optional[list[str]] = None,
        chunks_lookup: Optional[dict[str, dict]] = None,
    ) -> list[SearchResult]:
        """
        Perform local hybrid search: ChromaDB (semantic) + SQLite FTS5 (keyword).

        Parameters
        ----------
        query : str
            Natural language search query.
        filter_types : list[str] | None
            Optional page_type filter (not yet implemented for local).
        chunks_lookup : dict | None
            Optional {chunk_id: chunk_dict} for text hydration.
            If not provided, relies on ChromaDB metadata.

        Returns
        -------
        list[SearchResult]
            Top-k results sorted by RRF score.
        """
        semantic_results = self._semantic_search(query)
        keyword_results = self._keyword_search(query)

        logger.info(
            f"Query: '{query[:50]}' → "
            f"{len(semantic_results)} semantic, {len(keyword_results)} keyword"
        )

        merged = self._rrf_merge(
            semantic_results,
            keyword_results,
            chunks_lookup or {},
        )

        logger.info(f"RRF merged → {len(merged)} results")
        for i, r in enumerate(merged[:3]):
            logger.info(
                f"  #{i + 1}: {r.page_title} > {r.section_heading} "
                f"(rrf={r.rrf_score:.4f})"
            )

        return merged
