"""Quick diagnostic for SQLite FTS keyword search."""
import sqlite3
from pathlib import Path

DB = Path("data/sqlite_fts.db")
conn = sqlite3.connect(DB)
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", cur.fetchall())

cur.execute("SELECT COUNT(*) FROM chunks_fts")
print("FTS rows:", cur.fetchone()[0])

cur.execute("SELECT chunk_id, page_title, text FROM chunks_fts LIMIT 1")
row = cur.fetchone()
if row:
    print("Sample chunk_id:", row[0])
    print("Sample page_title:", row[1])
    print("Sample text (first 200):", row[2][:200] if row[2] else None)

# Test a simple FTS match
cur.execute("SELECT chunk_id, page_title FROM chunks_fts WHERE chunks_fts MATCH 'water' LIMIT 5")
rows = cur.fetchall()
print(f"\nFTS MATCH 'water': {len(rows)} results")
for r in rows:
    print(" ", r)

conn.close()
