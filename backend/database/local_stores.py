"""
Local database stores for the Ore-acle offline pipeline.

ChromaDB  → vector (semantic) search
SQLite FTS5 → keyword (BM25-like) search

Both are populated by the pipeline orchestrator and queried at
retrieval time by the local search module.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np

from backend.config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ChromaDB (Semantic)
# ---------------------------------------------------------------------------

class ChromaStore:
    """Manages a ChromaDB collection for semantic search.

    Each document is a chunk. Embeddings are supplied externally
    (from EmbedderProtocol) — ChromaDB is used as a pure vector index.

    Pass *embedding_model* to create a model-specific collection
    (different embedding models produce different dimensions).
    """

    DEFAULT_COLLECTION = "ore_acle_chunks"

    def __init__(
        self,
        db_dir: Optional[Path] = None,
        embedding_model: Optional[str] = None,
    ):
        self.db_dir = str(db_dir or settings.chroma_db_dir)
        self._collection_name = self._make_collection_name(embedding_model)
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    @classmethod
    def _make_collection_name(cls, model_id: Optional[str]) -> str:
        if model_id is None:
            return cls.DEFAULT_COLLECTION
        safe = model_id.replace("/", "_").replace("-", "_").replace(".", "_")
        return f"chunks_{safe}"

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.db_dir)
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def ingest(
        self,
        chunk_ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict],
        batch_size: int = 500,
    ) -> int:
        """Upsert chunks into ChromaDB in batches.

        Parameters
        ----------
        chunk_ids : list[str]
            Unique chunk identifiers.
        embeddings : np.ndarray
            Shape ``(n, dim)`` float32 embeddings.
        metadatas : list[dict]
            Per-chunk metadata dicts. Complex values (lists/dicts) are
            JSON-serialised to strings before storage.
        batch_size : int
            ChromaDB upsert batch size (keep ≤5000).

        Returns
        -------
        int
            Number of chunks ingested.
        """
        # Deduplicate by chunk_id, keeping the first occurrence.
        # ChromaDB raises DuplicateIDError if the same ID appears twice in one batch.
        seen: set = set()
        unique_ids: list = []
        unique_embs: list = []
        unique_metas: list = []
        for cid, emb, meta in zip(chunk_ids, embeddings, metadatas):
            if cid not in seen:
                seen.add(cid)
                unique_ids.append(cid)
                unique_embs.append(emb)
                unique_metas.append(meta)
        n_dupes = len(chunk_ids) - len(unique_ids)
        if n_dupes:
            logger.warning(f"Dropped {n_dupes} duplicate chunk IDs before ChromaDB upsert")
        chunk_ids = unique_ids
        embeddings = np.stack(unique_embs, axis=0)
        metadatas = unique_metas

        n = len(chunk_ids)
        assert n == embeddings.shape[0] == len(metadatas)

        # ChromaDB metadata values must be str | int | float | bool.
        # Serialise complex fields.
        clean_metas = []
        for m in metadatas:
            clean = {}
            for k, v in m.items():
                if isinstance(v, (list, dict)):
                    clean[k] = json.dumps(v, ensure_ascii=False)
                elif v is None:
                    clean[k] = ""
                else:
                    clean[k] = v
            clean_metas.append(clean)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            self.collection.upsert(
                ids=chunk_ids[start:end],
                embeddings=embeddings[start:end].tolist(),
                metadatas=clean_metas[start:end],
            )
            logger.info(f"ChromaDB upsert {end}/{n}")

        logger.info(f"ChromaDB ingestion complete: {n} chunks")
        return n

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 20,
    ) -> list[dict]:
        """Semantic nearest-neighbour search.

        Returns list of dicts with keys: id, distance, metadata.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["metadatas", "distances"],
        )
        out = []
        for i, cid in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            # Deserialise JSON fields
            for key in ("images", "infobox", "related_pages", "categories"):
                if key in meta and isinstance(meta[key], str):
                    try:
                        meta[key] = json.loads(meta[key])
                    except (json.JSONDecodeError, TypeError):
                        pass
            out.append({
                "id": cid,
                "distance": results["distances"][0][i],
                **meta,
            })
        return out

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection."""
        try:
            self.client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = None
        _ = self.collection  # recreate


# ---------------------------------------------------------------------------
# SQLite FTS5 (Keyword)
# ---------------------------------------------------------------------------

class SQLiteStore:
    """Manages a SQLite FTS5 virtual table for keyword search.

    Stores chunk_id, page_title, section_heading, and full text.
    BM25 ranking is handled natively by FTS5.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or settings.sqlite_db_path)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._ensure_tables()
        return self._conn

    def _ensure_tables(self) -> None:
        """Create the FTS5 virtual table if it doesn't exist."""
        self.conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id,
                page_title,
                section_heading,
                text,
                tokenize='porter unicode61'
            );
        """)

    def ingest(self, chunks: list[dict], batch_size: int = 1000) -> int:
        """Insert chunks into the FTS5 table.

        Parameters
        ----------
        chunks : list[dict]
            Each dict must have: chunk_id, page_title, section_heading, text.
        batch_size : int
            SQLite batch size for executemany.

        Returns
        -------
        int
            Number of rows inserted.
        """
        rows = [
            (
                c["chunk_id"],
                c.get("page_title", ""),
                c.get("section_heading", ""),
                c.get("text", ""),
            )
            for c in chunks
        ]
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            self.conn.executemany(
                "INSERT INTO chunks_fts(chunk_id, page_title, section_heading, text) "
                "VALUES (?, ?, ?, ?)",
                rows[start:end],
            )
            logger.info(f"SQLite FTS insert {end}/{len(rows)}")

        self.conn.commit()
        logger.info(f"SQLite FTS ingestion complete: {len(rows)} chunks")
        return len(rows)

    def get_by_ids(self, chunk_ids: list[str]) -> dict[str, dict]:
        """Fetch chunk text and metadata by chunk_id.

        Returns a {chunk_id: {"text": ..., "page_title": ..., "section_heading": ...}} dict.
        Only IDs that exist in the table are returned.
        """
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        cursor = self.conn.execute(
            f"SELECT chunk_id, page_title, section_heading, text "
            f"FROM chunks_fts WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        return {row["chunk_id"]: dict(row) for row in cursor.fetchall()}

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Keyword search using FTS5 BM25 ranking.

        Returns list of dicts with keys: chunk_id, page_title,
        section_heading, rank (lower = better match).
        """
        # Strip FTS5 special characters that cause syntax errors
        import re as _re
        safe_query = _re.sub(r'[^\w\s]', ' ', query, flags=_re.UNICODE).strip()
        if not safe_query:
            return []
        # Use OR between terms so natural-language queries return results.
        # Filter out terms ≤ 2 chars (stop words like "a", "i", "in", "do").
        terms = [t for t in safe_query.split() if len(t) > 2]
        if not terms:
            return []
        fts_query = " OR ".join(terms)
        cursor = self.conn.execute(
            """
            SELECT chunk_id, page_title, section_heading, rank
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def count(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM chunks_fts"
        ).fetchone()
        return row[0] if row else 0

    def reset(self) -> None:
        """Drop and recreate the FTS table."""
        self.conn.executescript("DROP TABLE IF EXISTS chunks_fts;")
        self._ensure_tables()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
