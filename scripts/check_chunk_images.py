"""Analyze image distribution in chunks.json and URL format."""
import json

chunks = json.load(open("data/processed/chunks.json", encoding="utf-8"))
with_images = [c for c in chunks if c.get("images")]

print(f"Total chunks: {len(chunks)}")
print(f"Chunks with images: {len(with_images)}")

if with_images:
    total_imgs = sum(len(c["images"]) for c in with_images)
    print(f"Total image refs: {total_imgs}")
    print(f"Avg images per chunk (when present): {total_imgs/len(with_images):.1f}")
    
    sample = with_images[0]["images"][0]
    print(f"\nSample image entry:")
    print(json.dumps(sample, indent=2))
    
    # Check URL patterns
    urls = set()
    for c in with_images[:100]:
        for img in c["images"]:
            urls.add(img.get("url", "")[:60])
    print(f"\nSample URLs (first 60 chars):")
    for u in list(urls)[:10]:
        print(f"  {u}")
