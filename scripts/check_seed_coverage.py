"""Check which seed questions are missing from the golden questionset."""
import json

with open("data/eval/seed_questions.json", encoding="utf-8") as f:
    seeds = json.load(f)

with open("data/eval/questionset.json", encoding="utf-8") as f:
    golden = json.load(f)

golden_pages = {q.get("source_page", "").strip().lower() for q in golden}
seed_pages_raw = {s.get("source_page", "").strip() for s in seeds}
seed_pages = {p.lower() for p in seed_pages_raw}

missing_pages = seed_pages_raw - {q.get("source_page", "").strip() for q in golden}
# case-insensitive missing
missing_pages = {p for p in seed_pages_raw if p.lower() not in golden_pages}
print(f"Seed pages   : {len(seed_pages_raw)}")
print(f"Golden pages : {len(golden_pages)}")
print(f"Missing pages: {sorted(missing_pages)}")
print()

golden_questions = {q.get("question", "").strip().lower() for q in golden}
missing_qs = [s for s in seeds if s.get("question", "").strip().lower() not in golden_questions]

print(f"Seed questions not present in golden ({len(missing_qs)}/{len(seeds)}):")
for s in missing_qs:
    page = s.get("source_page", "?")
    q = s.get("question", "?")
    diff = s.get("difficulty", "?")
    print(f"  [{diff}] [{page}] {q}")
