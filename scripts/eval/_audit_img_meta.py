import json

data = json.load(open("data/processed/metadata.json", encoding="utf-8"))
pages = data["pages"]
print(f"total pages: {len(pages)}")

count = 0
for page in pages:
    for img in page.get("images", []):
        if img.get("alt_text") or img.get("caption"):
            print(f"\n--- page: {page.get('title','')[:60]}")
            for k in ("local_filename", "section", "alt_text", "caption", "surrounding_text"):
                v = str(img.get(k, ""))
                print(f"  {k}: {v[:120]}")
            count += 1
            if count >= 10:
                break
    if count >= 10:
        break
