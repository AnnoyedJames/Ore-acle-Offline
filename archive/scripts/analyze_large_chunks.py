"""Deep-dive into oversized chunks to understand why they weren't split."""
import json

with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# --- LARGE CHUNKS (>512 tokens) ---
large = sorted([c for c in chunks if c["token_count"] > 512], key=lambda x: -x["token_count"])

print("=== Chunks > 8192 tokens (will be truncated by BGE-M3) ===")
over_limit = [c for c in large if c["token_count"] > 8192]
print(f"Count: {len(over_limit)}")
for c in over_limit:
    title = c["page_title"][:50].ljust(50)
    section = c["section_heading"][:40]
    # Check if it's a table — show first 200 chars to see structure
    text_preview = c["text"][:200].replace("\n", "\\n")
    print(f"  {c['token_count']:>6} | {c['chunk_type']:>8} | {title} | {section}")
    print(f"         preview: {text_preview}")
    print()

print("\n=== Chunks 2048-8192 tokens — breakdown by type ===")
mid_large = [c for c in large if 2048 <= c["token_count"] < 8192]
type_counts = {}
for c in mid_large:
    type_counts[c["chunk_type"]] = type_counts.get(c["chunk_type"], 0) + 1
print(f"Count: {len(mid_large)}")
for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {t}: {cnt}")

# Show some section-type large chunks
print("\n=== Sample large SECTION chunks (2048+) ===")
large_sections = [c for c in mid_large if c["chunk_type"] == "section"][:5]
for c in large_sections:
    title = c["page_title"][:50].ljust(50)
    section = c["section_heading"][:40]
    text_preview = c["text"][:300].replace("\n", "\\n")
    print(f"  {c['token_count']:>6} | {title} | {section}")
    print(f"         preview: {text_preview}")
    print()

# Show some table-type large chunks
print("=== Sample large TABLE chunks (2048+) ===")
large_tables = [c for c in mid_large if c["chunk_type"] == "table"][:5]
for c in large_tables:
    title = c["page_title"][:50].ljust(50)
    section = c["section_heading"][:40]
    # Count rows in the table
    lines = c["text"].split("\n")
    data_rows = len([l for l in lines if l.startswith("|") and not l.startswith("|---")])
    print(f"  {c['token_count']:>6} | {title} | {section}")
    print(f"         rows: {data_rows}, lines: {len(lines)}")
    # Show first 2 rows to see column width
    for line in lines[:3]:
        print(f"         {line[:150]}")
    print()

# --- SMALL CHUNKS ---
print("\n=== Small chunks (<10 tokens) by page ===")
small = [c for c in chunks if c["token_count"] < 10]
page_groups = {}
for c in small:
    key = c["page_title"]
    page_groups.setdefault(key, []).append(c)

for page, page_chunks in sorted(page_groups.items(), key=lambda x: -len(x[1])):
    print(f"  {page}: {len(page_chunks)} tiny chunks")
    for c in page_chunks[:3]:
        print(f"    {c['token_count']} tokens | {c['chunk_type']:>8} | text: {repr(c['text'][:80])}")

# Also check: pages with many small chunks (10-30 tokens) that could be merged
print("\n=== Pages with 5+ chunks under 30 tokens ===")
under30 = [c for c in chunks if c["token_count"] < 30]
page_groups30 = {}
for c in under30:
    page_groups30.setdefault(c["page_title"], []).append(c)

for page, pcs in sorted(page_groups30.items(), key=lambda x: -len(x[1]))[:15]:
    if len(pcs) >= 5:
        types = {}
        for c in pcs:
            types[c["chunk_type"]] = types.get(c["chunk_type"], 0) + 1
        print(f"  {page}: {len(pcs)} chunks < 30 tokens | types: {types}")
