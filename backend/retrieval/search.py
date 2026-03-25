"""
Hybrid Search — Reciprocal Rank Fusion over semantic + keyword results.

Semantic search: queries Pinecone for nearest vectors with full metadata.
Keyword search: queries Supabase tsvector for matching chunk IDs + ranks.
For keyword-only results, hydrates metadata from Pinecone via fetch().
Merges results using RRF (k=60).

Architecture:
  - Pinecone: vectors + full chunk metadata (text, images, etc.)
  - Supabase: keyword search index only (tsvector, no text stored)
  - Hydration: always from Pinecone (not Supabase)

Usage:
    from retrieval.search import HybridSearch
    search = HybridSearch()
    results = search.search("How do I find diamonds?")
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for hybrid search."""
    # Supabase (keyword search index only)
    supabase_url: str = ""
    supabase_key: str = ""
    # Pinecone (semantic search + metadata hydration)
    pinecone_api_key: str = ""
    pinecone_index_name: str = "ore-acle"
    # Search params
    semantic_candidates: int = 20
    keyword_candidates: int = 20
    top_k: int = 10
    rrf_k: int = 60  # RRF constant
    # Embedding model for query encoding
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_device: str = "cuda"


@dataclass
class SearchResult:
    """A single search result with RRF score."""
    chunk_id: str
    page_title: str
    page_url: str
    section_heading: str
    section_level: int
    text: str
    token_count: int
    chunk_type: str
    page_type: str
    infobox: Optional[dict]
    images: list
    rrf_score: float
    # Source scores for debugging
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None


class HybridSearch:
    """
    Hybrid search combining semantic (Pinecone) and keyword (Supabase tsvector)
    retrieval using Reciprocal Rank Fusion.

    RRF formula: score(d) = Σ 1/(k + rank_i(d))
    where k=60 (default) and rank_i is the rank from each retrieval method.

    Pinecone is the source of truth for chunk data.
    Supabase provides keyword search but stores no text.
    """

    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self._supabase = None
        self._pinecone_index = None
        self.embedder = None

    def _init_supabase(self):
        """Initialize Supabase client."""
        if self._supabase is not None:
            return

        from supabase import create_client

        url = self.config.supabase_url
        key = self.config.supabase_key

        if not url or not key:
            from config.settings import settings
            url = url or settings.supabase_url
            key = key or settings.supabase_service_key

        if not url or not key:
            raise ValueError("Supabase credentials not configured")

        self._supabase = create_client(url, key)

    def _init_pinecone(self):
        """Initialize Pinecone client."""
        if self._pinecone_index is not None:
            return

        from pinecone import Pinecone

        api_key = self.config.pinecone_api_key
        if not api_key:
            from config.settings import settings
            api_key = settings.pinecone_api_key

        if not api_key:
            raise ValueError("Pinecone API key not configured")

        pc = Pinecone(api_key=api_key)
        self._pinecone_index = pc.Index(self.config.pinecone_index_name)

    def _init_embedder(self):
        """Lazy-load the embedding model for query encoding."""
        if self.embedder is not None:
            return

        from embeddings.generator import EmbeddingGenerator, EmbeddingConfig

        config = EmbeddingConfig(
            model_name=self.config.embedding_model,
            device=self.config.embedding_device,
        )
        self.embedder = EmbeddingGenerator(config)

    def _embed_query(self, query: str) -> list[float]:
        """Embed a query using Nomic v1.5 with search_query prefix."""
        self._init_embedder()
        embedding = self.embedder.embed_query(query)
        return embedding.tolist()

    def _parse_pinecone_metadata(self, metadata: dict) -> dict:
        """Parse Pinecone metadata back into a chunk dict.

        Complex fields (images, infobox) are stored as JSON strings
        in Pinecone metadata and need to be deserialized.
        """
        images_json = metadata.get("images_json", "[]")
        infobox_json = metadata.get("infobox_json", "")

        return {
            "page_title": metadata.get("page_title", ""),
            "page_url": metadata.get("page_url", ""),
            "section_heading": metadata.get("section_heading", ""),
            "section_level": metadata.get("section_level", 2),
            "text": metadata.get("text", ""),
            "token_count": metadata.get("token_count", 0),
            "chunk_type": metadata.get("chunk_type", "section"),
            "page_type": metadata.get("page_type", "other"),
            "images": json.loads(images_json) if images_json else [],
            "infobox": json.loads(infobox_json) if infobox_json else None,
        }

    def _semantic_search(
        self, query_embedding: list[float], filter_types: Optional[list[str]] = None
    ) -> list[dict]:
        """
        Query Pinecone for nearest vectors with full metadata.

        Returns list of chunk dicts with a 'similarity' field.
        """
        self._init_pinecone()

        # Build Pinecone filter if page types specified
        pc_filter = None
        if filter_types:
            pc_filter = {"page_type": {"$in": filter_types}}

        # Query Pinecone with metadata included
        results = self._pinecone_index.query(
            vector=query_embedding,
            top_k=self.config.semantic_candidates,
            filter=pc_filter,
            include_metadata=True,  # Full metadata from Pinecone
        )

        if not results.matches:
            return []

        enriched = []
        for m in results.matches:
            chunk = self._parse_pinecone_metadata(m.metadata or {})
            chunk["id"] = m.id
            chunk["similarity"] = m.score
            enriched.append(chunk)

        return enriched

    def _keyword_search(
        self, query: str, filter_types: Optional[list[str]] = None
    ) -> list[dict]:
        """Call the Supabase match_chunks_keyword RPC function.

        Returns slim results: id, page_title, page_url, section_heading,
        page_type, rank. No text — Supabase is index-only.
        """
        self._init_supabase()

        params = {
            "search_query": query,
            "match_count": self.config.keyword_candidates,
        }
        if filter_types:
            params["filter_types"] = filter_types

        result = self._supabase.rpc("match_chunks_keyword", params).execute()
        return result.data or []

    def _hydrate_from_pinecone(self, ids: list[str]) -> dict[str, dict]:
        """Fetch full metadata for specific chunk IDs from Pinecone.

        Used to hydrate keyword-only results that aren't in the
        semantic search results.
        """
        if not ids:
            return {}

        self._init_pinecone()
        response = self._pinecone_index.fetch(ids=ids)

        result = {}
        for vid, vector in response.vectors.items():
            chunk = self._parse_pinecone_metadata(vector.metadata or {})
            chunk["id"] = vid
            result[vid] = chunk

        return result

    def _rrf_merge(
        self,
        semantic_results: list[dict],
        keyword_results: list[dict],
    ) -> list[SearchResult]:
        """
        Merge semantic and keyword results using Reciprocal Rank Fusion.

        RRF score for document d = Σ 1/(k + rank_i(d))
        """
        k = self.config.rrf_k
        scores: dict[str, float] = {}
        chunks: dict[str, dict] = {}
        semantic_scores: dict[str, float] = {}
        keyword_scores: dict[str, float] = {}

        # Score semantic results (these have full metadata)
        for rank, result in enumerate(semantic_results):
            cid = result["id"]
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
            chunks[cid] = result
            semantic_scores[cid] = result.get("similarity", 0)

        # Score keyword results (slim — may lack text)
        for rank, result in enumerate(keyword_results):
            cid = result["id"]
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
            if cid not in chunks:
                chunks[cid] = result  # Keyword-only, will need hydration
            keyword_scores[cid] = result.get("rank", 0)

        # Hydrate keyword-only results from Pinecone
        semantic_ids = {r["id"] for r in semantic_results}
        keyword_only_ids = [cid for cid in scores if cid not in semantic_ids]
        if keyword_only_ids:
            hydrated = self._hydrate_from_pinecone(keyword_only_ids)
            for cid, data in hydrated.items():
                chunks[cid] = data

        # Sort by RRF score descending
        sorted_ids = sorted(scores.keys(), key=lambda x: -scores[x])
        top_ids = sorted_ids[: self.config.top_k]

        results = []
        for cid in top_ids:
            c = chunks[cid]
            # Skip if we couldn't hydrate (missing from Pinecone)
            if "text" not in c:
                logger.warning(f"Chunk {cid} has no text data, skipping")
                continue
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
                infobox=c.get("infobox"),
                images=c.get("images", []),
                rrf_score=scores[cid],
                semantic_score=semantic_scores.get(cid),
                keyword_score=keyword_scores.get(cid),
            ))

        return results

    def search(
        self,
        query: str,
        filter_types: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """
        Perform hybrid search: semantic (Pinecone) + keyword (Supabase) with RRF fusion.

        Args:
            query: Natural language search query
            filter_types: Optional list of page_types to filter by

        Returns:
            Top-k SearchResult objects sorted by RRF score
        """
        # 1. Embed query
        query_embedding = self._embed_query(query)

        # 2. Run both searches
        semantic_results = self._semantic_search(query_embedding, filter_types)
        keyword_results = self._keyword_search(query, filter_types)

        logger.info(
            f"Query: '{query[:50]}...' → "
            f"{len(semantic_results)} semantic, {len(keyword_results)} keyword"
        )

        # 3. Merge with RRF
        merged = self._rrf_merge(semantic_results, keyword_results)

        logger.info(f"RRF merged → {len(merged)} results")
        for i, r in enumerate(merged[:3]):
            logger.info(
                f"  #{i + 1}: {r.page_title} > {r.section_heading} "
                f"(rrf={r.rrf_score:.4f})"
            )

        return merged
