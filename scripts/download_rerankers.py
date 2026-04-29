"""Download the three reranker models to the HF cache, one at a time."""
import os

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

from huggingface_hub import snapshot_download

models = [
    ("BAAI/bge-reranker-v2-m3", "~2.3 GB"),
    ("Qwen/Qwen3-Reranker-4B", "~8 GB"),
    ("zeroentropy/zerank-2", "~8 GB"),
]

for i, (model_id, size) in enumerate(models, 1):
    print(f"\n--- [{i}/{len(models)}] Downloading {model_id} ({size}) ---", flush=True)
    snapshot_download(model_id)
    print(f"--- Done: {model_id} ---", flush=True)

print("\n=== All models downloaded. Ready to run eval. ===")
