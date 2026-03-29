"""
Pipeline Orchestrator — end-to-end data processing for Ore-acle Offline.

Runs each stage sequentially, reusing intermediate artifacts when possible.

Stages:
  1. Scrape   → data/raw/html/*.html  (skip if HTML already exists)
  2. Download → data/raw/images/*.webp (skip if images already exist)
  3. Clean    → data/processed/metadata.json
  4. Chunk    → data/processed/chunks.json
  5. Embed    → data/processed/embeddings/{embeddings.npy, chunk_ids.json}
  6. Ingest   → ChromaDB + SQLite FTS5

Usage:
    python -m backend.pipeline.run                # full pipeline
    python -m backend.pipeline.run --from embed   # resume from embedding stage
    python -m backend.pipeline.run --only ingest  # run a single stage
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from backend.config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

STAGES = ["scrape", "download", "clean", "chunk", "embed", "ingest"]


def _stage_scrape() -> None:
    """Stage 1: Scrape Minecraft Wiki HTML pages."""
    from backend.scraper.wiki_scraper import MinecraftWikiScraper

    scraper = MinecraftWikiScraper()
    scraper.run()


def _stage_download() -> None:
    """Stage 2: Download images from local HTML files."""
    from backend.scraper.image_downloader import ImageDownloader

    downloader = ImageDownloader()
    downloader.process_html_files()


def _stage_clean() -> None:
    """Stage 3: Clean HTML → structured metadata.json."""
    from backend.preprocessing.text_cleaner import TextCleaner

    cleaner = TextCleaner()
    cleaner.process_all()


def _stage_chunk() -> None:
    """Stage 4: Chunk cleaned pages → chunks.json."""
    from backend.preprocessing.chunker import Chunker

    chunker = Chunker()
    chunker.chunk_all()


def _stage_embed() -> None:
    """Stage 5: Generate embeddings for all chunks."""
    from backend.embeddings.generator import EmbeddingGenerator

    generator = EmbeddingGenerator()
    embeddings, ids = generator.generate(resume=True)
    logger.info(f"Embed: generated {len(ids)} embeddings, shape {embeddings.shape}")


def _stage_ingest() -> None:
    """Stage 6: Load chunks + embeddings into ChromaDB and SQLite FTS5."""
    from backend.database.local_stores import ChromaStore, SQLiteStore

    # Load chunks
    chunks_path = settings.data_processed_dir / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.json not found at {chunks_path}. Run 'chunk' stage first.")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"Ingest: loaded {len(chunks)} chunks")

    # Load embeddings
    emb_path = settings.data_processed_dir / "embeddings" / "embeddings.npy"
    ids_path = settings.data_processed_dir / "embeddings" / "chunk_ids.json"
    if not emb_path.exists() or not ids_path.exists():
        raise FileNotFoundError("Embeddings not found. Run 'embed' stage first.")
    embeddings = np.load(emb_path)
    with open(ids_path, "r") as f:
        chunk_ids = json.load(f)
    logger.info(f"Ingest: loaded {len(chunk_ids)} embeddings, shape {embeddings.shape}")

    # Build chunk_id → chunk dict for metadata
    chunk_map = {c["chunk_id"]: c for c in chunks}

    # Ensure embedding IDs match chunk IDs
    missing = [cid for cid in chunk_ids if cid not in chunk_map]
    if missing:
        logger.warning(f"Ingest: {len(missing)} embedding IDs not found in chunks.json")

    # --- ChromaDB ---
    chroma = ChromaStore()
    if chroma.count() >= len(chunk_ids):
        logger.info(f"ChromaDB: already has {chroma.count()} entries, skipping.")
    else:
        logger.info("ChromaDB: ingesting...")
        if chroma.count() > 0:
            chroma.reset()

        # Prepare metadata for ChromaDB (subset of chunk fields)
        metadatas = []
        for cid in chunk_ids:
            c = chunk_map.get(cid, {})
            metadatas.append({
                "page_title": c.get("page_title", ""),
                "page_url": c.get("page_url", ""),
                "section_heading": c.get("section_heading", ""),
                "section_level": c.get("section_level", 2),
                "text": c.get("text", ""),
                "token_count": c.get("token_count", 0),
                "chunk_type": c.get("chunk_type", "section"),
                "page_type": c.get("page_type", "other"),
                "images": c.get("images", []),
                "infobox": c.get("infobox") or {},
            })

        chroma.ingest(chunk_ids, embeddings, metadatas)

    # --- SQLite FTS5 ---
    sqlite = SQLiteStore()
    if sqlite.count() >= len(chunks):
        logger.info(f"SQLite FTS: already has {sqlite.count()} entries, skipping.")
    else:
        logger.info("SQLite FTS: ingesting...")
        if sqlite.count() > 0:
            sqlite.reset()
        sqlite.ingest(chunks)

    sqlite.close()
    logger.info("Ingest: complete!")


STAGE_FUNCS = {
    "scrape": _stage_scrape,
    "download": _stage_download,
    "clean": _stage_clean,
    "chunk": _stage_chunk,
    "embed": _stage_embed,
    "ingest": _stage_ingest,
}


def run_pipeline(
    from_stage: str | None = None,
    only_stage: str | None = None,
) -> None:
    """Run the data processing pipeline.

    Parameters
    ----------
    from_stage : str | None
        Start from this stage (inclusive). Earlier stages are skipped.
    only_stage : str | None
        Run only this single stage.
    """
    if only_stage:
        if only_stage not in STAGE_FUNCS:
            raise ValueError(f"Unknown stage: {only_stage}. Choose from {STAGES}")
        stages_to_run = [only_stage]
    elif from_stage:
        if from_stage not in STAGE_FUNCS:
            raise ValueError(f"Unknown stage: {from_stage}. Choose from {STAGES}")
        idx = STAGES.index(from_stage)
        stages_to_run = STAGES[idx:]
    else:
        stages_to_run = STAGES

    logger.info(f"Pipeline: running stages {stages_to_run}")
    t0 = time.time()

    for stage_name in stages_to_run:
        logger.info(f"=== Stage: {stage_name} ===")
        ts = time.time()
        STAGE_FUNCS[stage_name]()
        elapsed = time.time() - ts
        logger.info(f"=== {stage_name} done ({elapsed:.1f}s) ===\n")

    total = time.time() - t0
    logger.info(f"Pipeline complete in {total:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ore-acle Offline Pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--from",
        dest="from_stage",
        choices=STAGES,
        help="Start from this stage (skip earlier ones)",
    )
    group.add_argument(
        "--only",
        dest="only_stage",
        choices=STAGES,
        help="Run only this stage",
    )
    args = parser.parse_args()

    run_pipeline(from_stage=args.from_stage, only_stage=args.only_stage)


if __name__ == "__main__":
    main()
