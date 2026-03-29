"""Smoke-test the pipeline stages clean → chunk → embed on 5 pages."""
import sys
import json
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# --- Stage: Clean (5 pages) ---
print("=== CLEAN ===")
from backend.preprocessing.text_cleaner import TextCleaner, CleanerConfig

config = CleanerConfig()
config.html_dir = Path("data/raw/html")
cleaner = TextCleaner(config)

# Pick 5 known pages
test_pages = ["Diamond.html", "Water.html", "Iron_Ingot.html", "Creeper.html", "Crafting.html"]
pages = []
for fname in test_pages:
    p = config.html_dir / fname
    if p.exists():
        result = cleaner.process_single(p)
        if result:
            pages.append(result)
            print(f"  Cleaned: {result.get('title', fname)} -> {len(result.get('sections', []))} sections")

print(f"Total pages cleaned: {len(pages)}")
if not pages:
    print("No pages found! Check HTML files.")
    sys.exit(1)

# --- Stage: Chunk ---
print("\n=== CHUNK ===")
from backend.preprocessing.chunker import Chunker

chunker = Chunker()
chunks = chunker.chunk_pages(pages)
print(f"Total chunks: {len(chunks)}")
if chunks:
    print(f"  Sample: {chunks[0]['chunk_id']} ({chunks[0]['token_count']} tokens)")

# --- Stage: Embed (first 10 chunks) ---
print("\n=== EMBED ===")
from backend.embeddings.generator import EmbeddingGenerator

gen = EmbeddingGenerator()
texts = [c["text"][:500] for c in chunks[:10]]  # limit text length for speed
embeddings = gen.embed_passages(texts)
print(f"Embedded {len(texts)} chunks -> shape {embeddings.shape}")
print(f"Dimension: {gen.dimension}")

# --- Stage: Ingest (into temp stores) ---
print("\n=== INGEST ===")
import tempfile
from backend.database.local_stores import ChromaStore, SQLiteStore
import numpy as np

tmpdir = tempfile.mkdtemp()
chroma = ChromaStore(db_dir=tmpdir)
chroma.ingest(
    chunk_ids=[c["chunk_id"] for c in chunks[:10]],
    embeddings=embeddings,
    metadatas=[{"page_title": c["page_title"], "text": c["text"][:200]} for c in chunks[:10]],
)
print(f"ChromaDB count: {chroma.count()}")

sqlite = SQLiteStore(db_path=":memory:")
sqlite.ingest(chunks[:10])
print(f"SQLite FTS count: {sqlite.count()}")

# --- Quick search test ---
print("\n=== SEARCH ===")
results = sqlite.search("diamond ore")
print(f"FTS5 results for 'diamond ore': {len(results)}")
for r in results[:3]:
    print(f"  {r['chunk_id']}: {r['page_title']} > {r['section_heading']}")

qvec = gen.embed_query("How do I find diamonds?")
sem_results = chroma.query(qvec, n_results=3)
print(f"Semantic results for 'How do I find diamonds?': {len(sem_results)}")
for r in sem_results[:3]:
    print(f"  {r['id']}: {r['page_title']} (dist={r['distance']:.4f})")

sqlite.close()
print("\n=== ALL SMOKE TESTS PASSED ===")
