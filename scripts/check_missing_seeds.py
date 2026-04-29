import json

with open("data/eval/questionset.json", encoding="utf-8") as f:
    golden = json.load(f)

for page in ["Hunger", "Chainmail Armor", "Nether portal"]:
    hits = [q for q in golden if q.get("source_page", "").lower() == page.lower()]
    print(f"{page}: {len(hits)} entries")
    for h in hits:
        print(f"  [{h['difficulty']}] {h['question'][:90]}")

print(f"\nTotal golden: {len(golden)}")
