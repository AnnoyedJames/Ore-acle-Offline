"""Quick script to test the ingest stage of the pipeline."""
import sys
import json
import time
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

print("=== Ingest Test ===")

# 1. Load chunks
t0 = time.time()
print("Loading chunks.json (1.1 GB)...")
with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"  {len(chunks)} chunks loaded in {time.time()-t0:.1f}s")

# 2. Load embeddings
t0 = time.time()
print("Loading embeddings...")
embeddings = np.load("data/processed/embeddings/embeddings.npy")
with open("data/processed/embeddings/chunk_ids.json", "r") as f:
    chunk_ids = json.load(f)
print(f"  {len(chunk_ids)} embeddings, shape {embeddings.shape} in {time.time()-t0:.1f}s")

# 3. Build lookup
chunk_map = {c["chunk_id"]: c for c in chunks}

# 4. Ingest into ChromaDB
from backend.database.local_stores import ChromaStore, SQLiteStore

chroma = ChromaStore()
print(f"ChromaDB current count: {chroma.count()}")

if chroma.count() < len(chunk_ids):
    if chroma.count() > 0:
        print("Resetting ChromaDB...")
        chroma.reset()

    print("Preparing metadata...")
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

    print(f"Ingesting {len(chunk_ids)} chunks into ChromaDB...")
    t0 = time.time()
    chroma.ingest(chunk_ids, embeddings, metadatas)
    print(f"  ChromaDB done in {time.time()-t0:.1f}s, count: {chroma.count()}")
else:
    print("ChromaDB already populated, skipping.")

# 5. Ingest into SQLite FTS5
sqlite = SQLiteStore()
print(f"SQLite FTS current count: {sqlite.count()}")

if sqlite.count() < len(chunks):
    if sqlite.count() > 0:
        print("Resetting SQLite FTS...")
        sqlite.reset()

    print(f"Ingesting {len(chunks)} chunks into SQLite FTS5...")
    t0 = time.time()
    sqlite.ingest(chunks)
    print(f"  SQLite done in {time.time()-t0:.1f}s, count: {sqlite.count()}")
else:
    print("SQLite FTS already populated, skipping.")

sqlite.close()
print("=== Done ===")
