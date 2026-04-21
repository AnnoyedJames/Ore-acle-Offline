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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np

from backend.config.settings import settings
from backend.database.local_stores import ChromaStore, SQLiteStore
from backend.embeddings.generator import EmbeddingGenerator
from dataclasses import dataclass, field

@dataclass
class SearchResult:
    chunk_id: str
    page_title: str
    page_url: str
    section_heading: str
    section_level: int
    text: str
    token_count: int
    chunk_type: str
    page_type: str
    rrf_score: float
    infobox: Optional[dict] = None
    images: list[dict] = field(default_factory=list)
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None

logger = logging.getLogger(__name__)


class HybridSearch:
    """
    Hybrid search combining semantic (ChromaDB) and keyword (SQLite FTS5)
    retrieval using Weighted Reciprocal Rank Fusion.

    Weighted RRF formula:
        score(d) = alpha * Σ 1/(k + rank_sem(d))
                 + (1 - alpha) * Σ 1/(k + rank_kw(d))

    Default: alpha=0.7 (semantic-heavy), k=20.
    Both parameters are overridable per-instance for ablation sweeps.
    """

    def __init__(
        self,
        chroma: Optional[ChromaStore] = None,
        sqlite: Optional[SQLiteStore] = None,
        embedder: Optional[EmbeddingGenerator] = None,
        rrf_alpha: Optional[float] = None,
        rrf_k: Optional[int] = None,
    ):
        self.chroma = chroma or ChromaStore()
        self.sqlite = sqlite or SQLiteStore()
        self._embedder = embedder
        # Search parameters from settings (can be overridden per-instance)
        self.semantic_candidates = settings.retrieval_semantic_candidates
        self.keyword_candidates = settings.retrieval_keyword_candidates
        self.top_k = settings.retrieval_top_k
        self.rrf_k = rrf_k if rrf_k is not None else settings.rrf_k
        self.rrf_alpha = rrf_alpha if rrf_alpha is not None else settings.rrf_alpha

    @property
    def embedder(self):
        if self._embedder is None:
            from backend.embeddings import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def _semantic_search(
        self,
        query: str,
        filter_types: Optional[list[str]] = None,
        query_vec: Optional["np.ndarray"] = None,
    ) -> list[dict]:
        """Embed query and search ChromaDB, optionally filtered by page_type.

        Parameters
        ----------
        query_vec : np.ndarray | None
            Pre-computed embedding vector. If provided, skips the embed_query
            API call (used by the eval to avoid per-question API round-trips).
        """
        if query_vec is None:
            query_vec = self.embedder.embed_query(query)
        results = self.chroma.query(query_vec, n_results=self.semantic_candidates,
                                    filter_page_types=filter_types or [])
        return results

    def _keyword_search(self, query: str) -> list[dict]:
        """Search SQLite FTS5 for matching chunks."""
        return self.sqlite.search(query, limit=self.keyword_candidates)

    def _rrf_merge(
        self,
        semantic_results: list[dict],
        keyword_results: list[dict],
        chunks_lookup: dict[str, dict],
        alpha_override: Optional[float] = None,
    ) -> list[SearchResult]:
        """Merge results using Weighted Reciprocal Rank Fusion.

        Semantic results are weighted by ``self.rrf_alpha``; keyword results
        by ``1 - self.rrf_alpha``.  Setting alpha=1.0 reduces to pure semantic
        reranking; 0.5 gives equal weight (standard RRF).
        """
        k = self.rrf_k
        alpha = alpha_override if alpha_override is not None else self.rrf_alpha
        scores: dict[str, float] = {}
        semantic_scores: dict[str, float] = {}
        keyword_scores: dict[str, float] = {}

        # Weighted RRF: semantic contributes alpha, keyword contributes (1 - alpha)
        for rank, result in enumerate(semantic_results):
            cid = result["id"]
            scores[cid] = scores.get(cid, 0) + alpha / (k + rank + 1)
            semantic_scores[cid] = 1.0 - result.get("distance", 0)

        # Score keyword results
        for rank, result in enumerate(keyword_results):
            cid = result["chunk_id"]
            scores[cid] = scores.get(cid, 0) + (1.0 - alpha) / (k + rank + 1)
            keyword_scores[cid] = abs(result.get("rank", 0))

        # Sort by RRF score descending
        sorted_ids = sorted(scores.keys(), key=lambda x: -scores[x])
        top_ids = sorted_ids[: self.top_k]

        # Build a lookup from all result sources in priority order:
        # 1. ChromaDB semantic metadata (has everything except text)
        # 2. chunks_lookup (optional in-memory dict, legacy)
        # 3. SQLite batch fetch (cheap, on-demand, only what we need)
        sem_by_id = {r["id"]: r for r in semantic_results}
        kw_by_id = {r["chunk_id"]: r for r in keyword_results}

        # Determine which IDs still need text hydration from SQLite
        ids_needing_text = [
            cid for cid in top_ids
            if cid not in chunks_lookup
            and not sem_by_id.get(cid, {}).get("text")
        ]
        sqlite_rows: dict[str, dict] = {}
        if ids_needing_text:
            sqlite_rows = self.sqlite.get_by_ids(ids_needing_text)

        results = []
        for cid in top_ids:
            # Prefer semantic result metadata (has full fields except text),
            # fall back to keyword result, then chunks_lookup, then SQLite
            c: dict = {}
            if cid in sem_by_id:
                c = sem_by_id[cid]
            elif cid in chunks_lookup:
                c = chunks_lookup[cid]
            elif cid in kw_by_id:
                c = kw_by_id[cid]

            # Hydrate text: prefer chunks_lookup / existing field, then SQLite
            text = c.get("text", "")
            if not text:
                if cid in chunks_lookup:
                    text = chunks_lookup[cid].get("text", "")
                elif cid in sqlite_rows:
                    text = sqlite_rows[cid].get("text", "")

            if not c and not text:
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

            token_count = c.get("token_count") or len(text) // 4

            results.append(SearchResult(
                chunk_id=cid,
                page_title=c.get("page_title", ""),
                page_url=c.get("page_url", "") or
                    f"https://minecraft.wiki/w/{c.get('page_title', '').replace(' ', '_')}",
                section_heading=c.get("section_heading", ""),
                section_level=c.get("section_level", 2),
                text=text,
                token_count=token_count,
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
        mode: str = "hybrid",
        filter_types: Optional[list[str]] = None,
        chunks_lookup: Optional[dict[str, dict]] = None,
        query_vec: Optional["np.ndarray"] = None,
    ) -> list[SearchResult]:
        """
        Perform local hybrid search: ChromaDB (semantic) + SQLite FTS5 (keyword).

        Parameters
        ----------
        query : str
            Natural language search query.
        mode : str
            ``"hybrid"`` (default), ``"semantic"``, or ``"keyword"``.
        filter_types : list[str] | None
            Optional list of ``page_type`` values to restrict results to
            (e.g. ``["mob", "item", "biome"]``). Applied to the semantic
            leg only; keyword results are post-filtered by page_type.
        chunks_lookup : dict | None
            Optional {chunk_id: chunk_dict} for text hydration.
            If not provided, relies on ChromaDB metadata.
        query_vec : np.ndarray | None
            Pre-computed query embedding. If supplied, skips the embed_query
            API call. Passed through to _semantic_search.

        Returns
        -------
        list[SearchResult]
            Top-k results sorted by RRF score.
        """
        _alpha_override: Optional[float] = None
        if mode == "semantic":
            semantic_results = self._semantic_search(query, filter_types, query_vec=query_vec)
            keyword_results = []
        elif mode == "keyword":
            semantic_results = []
            keyword_results = self._keyword_search(query)
        else:  # hybrid
            # Run semantic and keyword searches in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                sem_future = executor.submit(
                    self._semantic_search, query, filter_types, query_vec
                )
                kw_future = executor.submit(self._keyword_search, query)
                semantic_results = sem_future.result()
                keyword_results = kw_future.result()
            # Semantic quality gate: if the best cosine similarity is below
            # threshold the embedding likely missed — shift weight to keyword.
            if semantic_results:
                best_sim = 1.0 - min(r.get("distance", 0) for r in semantic_results)
                if best_sim < 0.55:
                    _alpha_override = 0.35
                    logger.info(
                        f"Semantic quality gate: best_sim={best_sim:.3f} < 0.55, "
                        f"alpha {self.rrf_alpha}→{_alpha_override}"
                    )
            keyword_results = self._keyword_search(query)

        logger.info(
            f"Query: '{query[:50]}' \u2192 "
            f"{len(semantic_results)} semantic, {len(keyword_results)} keyword"
        )

        merged = self._rrf_merge(
            semantic_results,
            keyword_results,
            chunks_lookup or {},
            alpha_override=_alpha_override if mode == "hybrid" else None,
        )

        # Post-filter keyword-only results by page_type if filter requested
        if filter_types and mode in ("keyword", "hybrid"):
            merged = [r for r in merged if r.page_type in filter_types]

        logger.info(f"RRF merged \u2192 {len(merged)} results")
        for i, r in enumerate(merged[:3]):
            logger.info(
                f"  #{i + 1}: {r.page_title} > {r.section_heading} "
                f"(rrf={r.rrf_score:.4f})"
            )

        # Build list of already-included chunk IDs
        existing_ids = {r.chunk_id for r in merged}

        # Auto-Crafting Retrieval: Search for related crafting recipes
        # Use a lightweight keyword query that targets the UI string we added
        recipe_query = query.lower().replace("how do you craft", "").replace("how do i craft", "").replace("how to craft", "").replace("recipe for", "").strip()
        if len(recipe_query) > 2:
            crafting_results = self._keyword_search(f'"[Crafting Recipe:" {recipe_query}')
            if crafting_results:
                # Get the top candidate that isn't already included
                new_recipes = [r for r in crafting_results if r["chunk_id"] not in existing_ids]
                if new_recipes:
                    top_recipe_chunk = new_recipes[0]
                    # We must run it through _rrf_merge for hydration, or manually hydrate
                    hydrated = self._rrf_merge([], [top_recipe_chunk], chunks_lookup or {}, alpha_override=0.0)
                    if hydrated:
                        hydrated[0].rrf_score = 0.0  # Pushed to end
                        merged.append(hydrated[0])
                        logger.info(f"Auto-injected crafting recipe: {hydrated[0].page_title}")

        return merged
