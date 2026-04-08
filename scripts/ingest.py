"""
Ingestion script: embeddings → ChromaDB & SQLite FTS5.

Generates embeddings (with resumable checkpointing) then populates:
  - ChromaDB  : model-specific collection with chunk metadata
  - SQLite FTS5: shared text index (all models share one DB)

Usage:
    # Default: BGE-M3, section_aware chunks
    python scripts/ingest.py

    # Specific embedding model
    python scripts/ingest.py --model nomic-ai/nomic-embed-text-v1.5

    # Skip SQLite (already ingested)
    python scripts/ingest.py --skip-sqlite

    # Wipe and re-ingest the ChromaDB collection for this model
    python scripts/ingest.py --reset-chroma

    # LangChain chunked variant (reads chunks_langchain.json)
    python scripts/ingest.py --chunks data/processed/chunks_langchain.json \
        --model BAAI/bge-m3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config.settings import EMBEDDING_MODELS, settings
from backend.database.local_stores import ChromaStore, SQLiteStore
from backend.embeddings.api_generator import ApiEmbeddingGenerator
from backend.embeddings.generator import EmbeddingConfig, EmbeddingGenerator

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metadata_for_chunk(c: dict) -> dict:
    """Extract ChromaDB-safe metadata fields from a chunk dict."""
    return {
        "page_title": c.get("page_title", ""),
        "page_url": c.get("page_url", ""),
        "section_heading": c.get("section_heading", ""),
        "section_level": c.get("section_level", 2),
        "token_count": c.get("token_count", 0),
        "chunk_type": c.get("chunk_type", "section"),
        "page_type": c.get("page_type", "other"),
        "text": c.get("text", ""),
        # images / infobox are complex — ChromaStore.ingest will JSON-serialise them
        "images": c.get("images", []),
        "infobox": c.get("infobox"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest chunks into ChromaDB (semantic) and SQLite (keyword).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default=settings.embedding_model,
        help=f"Embedding model ID (default: {settings.embedding_model}). "
             f"Must be a key in EMBEDDING_MODELS.",
    )
    parser.add_argument(
        "--chunking",
        choices=["section_aware", "langchain"],
        default="section_aware",
        help="Chunking strategy variant (default: section_aware). 'langchain' suffixes "
             "the ChromaDB collection name with '__langchain' and uses a separate SQLite DB.",
    )
    parser.add_argument(
        "--chunks",
        default=str(settings.chunks_file),
        help="Path to chunks JSON file (default: data/processed/chunks.json).",
    )
    parser.add_argument(
        "--skip-sqlite",
        action="store_true",
        help="Skip SQLite ingestion (useful when re-ingesting for a new embedding model).",
    )
    parser.add_argument(
        "--reset-chroma",
        action="store_true",
        help="Delete existing ChromaDB collection for this model before ingesting.",
    )
    parser.add_argument(
        "--reset-sqlite",
        action="store_true",
        help="Drop and recreate the SQLite FTS table before ingesting.",
    )
    args = parser.parse_args()

    model_id = args.model
    if model_id not in EMBEDDING_MODELS:
        logger.error(
            f"Unknown model '{model_id}'. Available: {list(EMBEDDING_MODELS.keys())}"
        )
        sys.exit(1)

    model_info = EMBEDDING_MODELS[model_id]

    # Determine chunking variant metadata
    CHUNKING_SQLITE = {
        "section_aware": settings.sqlite_db_path,
        "langchain": Path("data/sqlite_fts_langchain.db"),
    }
    CHUNKING_SUFFIX = {
        "section_aware": "",
        "langchain": "__langchain",
    }
    CHUNKING_DEFAULT_CHUNKS = {
        "section_aware": str(settings.chunks_file),
        "langchain": "data/processed/chunks_langchain.json",
    }

    # Override --chunks default if not explicitly provided and chunking != section_aware
    chunks_arg = args.chunks
    if chunks_arg == str(settings.chunks_file) and args.chunking != "section_aware":
        chunks_arg = CHUNKING_DEFAULT_CHUNKS[args.chunking]

    chunks_path = Path(chunks_arg)

    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Generate (or load cached) embeddings
    # ------------------------------------------------------------------
    logger.info(f"=== STEP 1: Embeddings for {model_id} ===")

    # Store embeddings in a chunking-variant subdir so they don't collide
    base_emb_dir = settings.get_model_embeddings_dir(model_id)
    suffix = CHUNKING_SUFFIX[args.chunking]
    output_dir = base_emb_dir.parent / (base_emb_dir.name + suffix) if suffix else base_emb_dir

    if model_info.backend == "api":
        generator = ApiEmbeddingGenerator(
            model_id=model_id,
            batch_size=settings.embedding_batch_size,
            output_dir=output_dir,
        )
    else:
        emb_config = EmbeddingConfig(
            chunks_file=chunks_path,
            model_name=model_id,
            batch_size=settings.embedding_batch_size,
            device=settings.embedding_device,
            truncate_dim=model_info.dimension,
            task_prefix=model_info.task_prefix,
            query_prefix=model_info.query_prefix,
        )
        emb_config.output_dir = output_dir
        generator = EmbeddingGenerator(config=emb_config)

    if model_info.backend == "api":
        # ApiEmbeddingGenerator.generate() accepts chunks_file explicitly
        embeddings, chunk_ids = generator.generate(chunks_file=chunks_path, resume=True)
    else:
        # EmbeddingGenerator reads chunks_file from its EmbeddingConfig
        embeddings, chunk_ids = generator.generate(resume=True)
    logger.info(f"Embeddings ready: {embeddings.shape}")

    # ------------------------------------------------------------------
    # 2. Load chunks for metadata
    # ------------------------------------------------------------------
    logger.info("=== STEP 2: Loading chunks for metadata ===")
    with open(chunks_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    # Build lookup by chunk_id to align with embedding order
    chunk_by_id = {c["chunk_id"]: c for c in all_chunks}
    ordered_chunks = [chunk_by_id[cid] for cid in chunk_ids]
    metadatas = [_metadata_for_chunk(c) for c in ordered_chunks]

    logger.info(f"Chunk metadata ready: {len(ordered_chunks)} entries")

    # ------------------------------------------------------------------
    # 3. Ingest into ChromaDB
    # ------------------------------------------------------------------
    collection_key = model_id + CHUNKING_SUFFIX[args.chunking]
    logger.info(f"=== STEP 3: ChromaDB ingest for collection key '{collection_key}' ===")
    chroma = ChromaStore(embedding_model=collection_key)

    if args.reset_chroma:
        logger.info("Resetting ChromaDB collection...")
        chroma.reset()

    existing = chroma.count()
    if existing > 0 and not args.reset_chroma:
        logger.info(
            f"ChromaDB collection already has {existing} docs. "
            "Pass --reset-chroma to re-ingest. Skipping."
        )
    else:
        ingested = chroma.ingest(
            chunk_ids=chunk_ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(f"ChromaDB: {ingested} chunks ingested")

    # ------------------------------------------------------------------
    # 4. Ingest into SQLite FTS5 (shared, model-agnostic)
    # ------------------------------------------------------------------
    if args.skip_sqlite:
        logger.info("=== STEP 4: SQLite skipped (--skip-sqlite) ===")
    else:
        logger.info("=== STEP 4: SQLite FTS5 ingest ===")
        sqlite = SQLiteStore(db_path=CHUNKING_SQLITE[args.chunking])

        if args.reset_sqlite:
            logger.info("Resetting SQLite FTS table...")
            sqlite.reset()

        existing_sql = sqlite.count()
        if existing_sql > 0 and not args.reset_sqlite:
            logger.info(
                f"SQLite already has {existing_sql} rows. "
                "Pass --reset-sqlite to re-ingest. Skipping."
            )
        else:
            ingested_sql = sqlite.ingest(all_chunks)
            logger.info(f"SQLite FTS5: {ingested_sql} chunks ingested")
        sqlite.close()

    logger.info("=== Ingestion complete ===")
    logger.info(f"  Model    : {model_id}")
    logger.info(f"  Chunking : {args.chunking}")
    logger.info(f"  ChromaDB collection: {chroma._collection_name}")
    logger.info(f"  ChromaDB dir       : {chroma.db_dir}")
    logger.info(f"  SQLite path        : {CHUNKING_SQLITE[args.chunking]}")


if __name__ == "__main__":
    main()
