"""Quick analysis of chunks.json to check quality and outliers."""
import json

with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Outliers: very large chunks
large = sorted([c for c in chunks if c["token_count"] > 1000], key=lambda x: -x["token_count"])[:10]
print("=== Top 10 largest chunks ===")
for c in large:
    title = c["page_title"][:40].ljust(40)
    section = c["section_heading"][:30]
    print(f"  {c['token_count']:>6} tokens | {c['chunk_type']:>8} | {title} | {section}")

# Very small chunks
small = [c for c in chunks if c["token_count"] <= 5]
print(f"\nChunks with <= 5 tokens: {len(small)}")
for c in small[:10]:
    title = c["page_title"][:40].ljust(40)
    print(f"  {c['token_count']:>2} tokens | {c['chunk_type']:>8} | {title} | text: {repr(c['text'][:60])}")

# Distribution
brackets = [(0, 10), (10, 50), (50, 100), (100, 256), (256, 512), (512, 1024), (1024, float("inf"))]
print("\n=== Token count distribution ===")
for lo, hi in brackets:
    count = len([c for c in chunks if lo <= c["token_count"] < hi])
    pct = count / len(chunks) * 100
    hi_str = str(int(hi)) if hi != float("inf") else "inf"
    print(f"  {lo}-{hi_str:>5}: {count:>6} ({pct:.1f}%)")
