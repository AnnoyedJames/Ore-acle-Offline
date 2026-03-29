"""Quick analysis of chunk storage breakdown by page_type."""
import json
from collections import Counter

with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Count by page_type
type_counts = Counter(c.get("page_type", "unknown") for c in chunks)

print(f"{'page_type':25s} | {'chunks':>8s} | {'text_MB':>8s} | {'emb_MB':>8s} | {'total_MB':>8s}")
print("-" * 70)

rows = []
for t, cnt in type_counts.most_common():
    text_mb = sum(len(c["text"]) for c in chunks if c.get("page_type") == t) / 1024 / 1024
    emb_mb = cnt * 768 * 4 / 1024 / 1024
    total = text_mb + emb_mb
    rows.append((t, cnt, text_mb, emb_mb, total))
    print(f"{t:25s} | {cnt:>8,} | {text_mb:>8.1f} | {emb_mb:>8.1f} | {total:>8.1f}")

print("-" * 70)
total_chunks = sum(r[1] for r in rows)
total_text = sum(r[2] for r in rows)
total_emb = sum(r[3] for r in rows)
total_all = sum(r[4] for r in rows)
print(f"{'TOTAL':25s} | {total_chunks:>8,} | {total_text:>8.1f} | {total_emb:>8.1f} | {total_all:>8.1f}")

# Show what keeping only game-content pages looks like
game_types = {"mob", "block", "item", "biome", "structure", "dimension",
              "enchantment", "effect", "mechanic", "entity", "tutorial"}
game_rows = [r for r in rows if r[0] in game_types]
gc = sum(r[1] for r in game_rows)
gt = sum(r[2] for r in game_rows)
ge = sum(r[3] for r in game_rows)
ga = sum(r[4] for r in game_rows)
print(f"\n--- Game-content only ({', '.join(sorted(game_types))}) ---")
print(f"{'GAME CONTENT':25s} | {gc:>8,} | {gt:>8.1f} | {ge:>8.1f} | {ga:>8.1f}")

# Show what gets cut
cut_types = {"version", "franchise", "media", "disambiguation", "redirect"}
cut_rows = [r for r in rows if r[0] in cut_types]
cc = sum(r[1] for r in cut_rows)
ct = sum(r[2] for r in cut_rows)
ce = sum(r[3] for r in cut_rows)
ca = sum(r[4] for r in cut_rows)
print(f"{'CUT (version/meta/etc)':25s} | {cc:>8,} | {ct:>8.1f} | {ce:>8.1f} | {ca:>8.1f}")

other_rows = [r for r in rows if r[0] not in game_types and r[0] not in cut_types]
oc = sum(r[1] for r in other_rows)
ot = sum(r[2] for r in other_rows)
oe = sum(r[3] for r in other_rows)
oa = sum(r[4] for r in other_rows)
print(f"{'UNCATEGORIZED (other)':25s} | {oc:>8,} | {ot:>8.1f} | {oe:>8.1f} | {oa:>8.1f}")

# Unique pages per type
page_counts = {}
for c in chunks:
    pt = c.get("page_type", "unknown")
    page_counts.setdefault(pt, set()).add(c.get("page_url", ""))
print(f"\n--- Unique pages per type ---")
for t, _ in type_counts.most_common():
    print(f"  {t:25s}: {len(page_counts[t]):>6,} pages")
