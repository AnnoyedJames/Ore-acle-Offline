"""
Nemotron VL Image Index Builder  (API version)
===============================================

Builds a semantic embedding index over all downloaded Minecraft Wiki images
using nvidia/llama-nemotron-embed-vl-1b-v2 via the OpenRouter embeddings API.
No local model weights are required.

Encoding strategy per image
----------------------------
- image + text mode when alt_text / caption / surrounding_text available
  (aggregated across all pages that reference the image, cap 3 contexts)
- image-only mode as fallback

Resumability
------------
The script checkpoints every CHECKPOINT_INTERVAL images.  If interrupted,
re-running the script will skip already-processed images and continue from
where it left off.

Checkpoint files (deleted on successful completion):
  data/eval/image_index_partial.npy
  data/eval/image_index_checkpoint.json

Final outputs
-------------
  data/eval/image_index_filenames.json   — ordered list[str] of local_filename
  data/eval/image_index_embeddings.npy   — float32 (N, 2048), L2-normalised
  data/eval/image_index_metadata.json    — {filename: description} for LLM tool responses

Usage
-----
    python scripts/eval/build_image_index.py
    python scripts/eval/build_image_index.py --batch-size 4 --delay 0.2
    python scripts/eval/build_image_index.py --limit 50    # smoke test
"""

import argparse
import base64
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_ID = "nvidia/llama-nemotron-embed-vl-1b-v2"
EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
EMBEDDING_DIM = 2048
CHECKPOINT_INTERVAL = 500  # save partial results every N images

OUT_DIR = Path("data/eval")
FILENAMES_PATH   = OUT_DIR / "image_index_filenames.json"
EMBEDDINGS_PATH  = OUT_DIR / "image_index_embeddings.npy"
METADATA_PATH    = OUT_DIR / "image_index_metadata.json"
CHECKPOINT_PATH  = OUT_DIR / "image_index_checkpoint.json"
PARTIAL_NPY_PATH = OUT_DIR / "image_index_partial.npy"


# ---------------------------------------------------------------------------
# Text aggregation
# ---------------------------------------------------------------------------

def _build_text_contexts(metadata_pages: list) -> dict:
    """
    Returns {local_filename: description_text} for use as the text side of
    image+text embeddings.  Only uses alt_text, caption, and surrounding_text
    (NOT section — section fields are often noisy nav-bar dumps or single
    words like 'Gallery' that add no semantic signal).
    Up to 3 unique contexts collected per image, joined with " | ".
    Images with no usable text fall back to image-only embedding mode.
    """
    contexts: dict = {}
    for page in metadata_pages:
        for img_ref in page.get("images", []):
            fname = img_ref.get("local_filename", "")
            if not fname:
                continue
            parts = []
            for field in ("alt_text", "caption"):
                v = (img_ref.get(field, "") or "").strip()
                if v:
                    parts.append(v)
            sur = (img_ref.get("surrounding_text", "") or "").strip()[:120]
            if sur:
                parts.append(sur)
            if not parts:
                continue
            text = " ".join(parts).strip()
            existing = contexts.setdefault(fname, [])
            if text not in existing and len(existing) < 3:
                existing.append(text)
    return {fname: " | ".join(texts) for fname, texts in contexts.items()}


def _build_lm_descriptions(metadata_pages: list) -> dict:
    """
    Returns {local_filename: short_description} for use in LLM tool responses.
    Picks the single best label per image: caption > alt_text >
    surrounding_text[:80] > filename stem.  Section is excluded.
    """
    best: dict = {}
    for page in metadata_pages:
        for img_ref in page.get("images", []):
            fname = img_ref.get("local_filename", "")
            if not fname or fname in best:
                continue
            for field in ("caption", "alt_text"):
                v = (img_ref.get(field, "") or "").strip()
                if v:
                    best[fname] = v
                    break
            if fname not in best:
                sur = (img_ref.get("surrounding_text", "") or "").strip()[:80]
                if sur:
                    best[fname] = sur
    return best


# ---------------------------------------------------------------------------
# OpenRouter embeddings API helper
# ---------------------------------------------------------------------------

def _api_embed_batch(
    inputs: list,
    api_key: str,
    delay: float,
    max_retries: int = 6,
) -> list:
    """
    POST a batch of inputs to the OpenRouter embeddings endpoint.
    Returns a list of embedding vectors (floats), one per input item,
    in the same order as inputs.
    Handles 429 / 5xx with exponential backoff.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_ID,
        "input": inputs,
        "encoding_format": "float",
    }

    backoff = 5.0
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                EMBEDDINGS_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                logger.warning(
                    f"  HTTP {resp.status_code} — backing off {backoff:.0f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
                continue
            resp.raise_for_status()
            data = resp.json()
            if "data" not in data:
                # API returned a non-standard response (e.g. upstream error body)
                logger.warning(
                    f"  Unexpected response (no 'data' key): {str(data)[:200]} "
                    f"— backing off {backoff:.0f}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
                continue
            # Sort by index to guarantee ordering matches input order
            items = sorted(data["data"], key=lambda x: x["index"])
            time.sleep(delay)
            return [item["embedding"] for item in items]
        except requests.RequestException as e:
            logger.warning(f"  Request error: {e} — backing off {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)

    raise RuntimeError(f"Embedding API failed after {max_retries} attempts")


def _make_input_item(fpath: str, description: str):
    """
    Build an OpenRouter embeddings input item for one image.
    Returns None if the image cannot be read.
    Uses image+text mode when description is available, image-only otherwise.
    """
    try:
        with open(fpath, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("utf-8")
    except Exception as e:
        logger.warning(f"  Cannot read {fpath}: {e}")
        return None

    content = []
    if description:
        content.append({"type": "text", "text": description})
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/webp;base64,{b64}"},
    })
    return {"content": content}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint():
    """
    Returns (already_processed_filenames, partial_embeddings_array).
    """
    if not CHECKPOINT_PATH.exists() or not PARTIAL_NPY_PATH.exists():
        return [], np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    try:
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            cp = json.load(f)
        processed = cp.get("processed_filenames", [])
        embs = np.load(PARTIAL_NPY_PATH)
        if len(processed) != len(embs):
            logger.warning("Checkpoint length mismatch — starting fresh.")
            return [], np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        logger.info(f"Resuming from checkpoint: {len(processed):,} images already done")
        return processed, embs
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e} — starting fresh")
        return [], np.empty((0, EMBEDDING_DIM), dtype=np.float32)


def _save_checkpoint(processed_filenames: list, embs: np.ndarray) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PARTIAL_NPY_PATH, embs)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"processed_filenames": processed_filenames}, f)


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_index(
    batch_size: int = 4,
    delay: float = 0.15,
    limit: int = None,
) -> None:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set in environment / .env")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load source data -------------------------------------------------
    logger.info("Loading image_metadata.json ...")
    with open("data/processed/image_metadata.json", encoding="utf-8") as f:
        raw_img_meta = json.load(f)["images"]

    filepath_lookup = {}
    for entry in raw_img_meta:
        fname = entry.get("local_filename", "")
        fpath = entry.get("file_path", "")
        if fname and fpath and Path(fpath).exists():
            filepath_lookup[fname] = fpath
    logger.info(f"  {len(filepath_lookup):,} images exist on disk")

    logger.info("Loading metadata.json ...")
    with open("data/processed/metadata.json", encoding="utf-8") as f:
        metadata_pages = json.load(f)["pages"]

    logger.info("Aggregating text contexts ...")
    text_contexts = _build_text_contexts(metadata_pages)
    with_text = sum(1 for f in filepath_lookup if f in text_contexts)
    logger.info(
        f"  {with_text:,} / {len(filepath_lookup):,} images have text "
        f"({with_text / max(len(filepath_lookup), 1) * 100:.1f}%)"
    )

    # ---- Build description metadata (for LLM tool responses) --------------
    logger.info("Building LLM descriptions ...")
    lm_descriptions = _build_lm_descriptions(metadata_pages)
    metadata_out = {}
    for fname in filepath_lookup:
        desc = lm_descriptions.get(fname, "")
        metadata_out[fname] = desc if desc else fname.replace("_", " ").rsplit(".", 1)[0]

    all_filenames = sorted(filepath_lookup.keys())
    if limit:
        all_filenames = all_filenames[:limit]
        logger.info(f"  Limited to {limit} images (--limit flag)")

    # ---- Resume from checkpoint if available ------------------------------
    done_filenames, done_embs = _load_checkpoint()
    done_set = set(done_filenames)
    remaining = [f for f in all_filenames if f not in done_set]
    logger.info(
        f"  {len(done_filenames):,} already done, "
        f"{len(remaining):,} remaining"
    )

    # Working buffers starting from checkpoint
    processed_filenames = list(done_filenames)
    all_embs = [done_embs[i] for i in range(len(done_embs))]

    n_failed = 0
    n = len(remaining)
    total = len(done_filenames) + n

    for batch_start in range(0, n, batch_size):
        batch_fnames = remaining[batch_start : batch_start + batch_size]

        inputs = []
        valid_fnames = []

        for fname in batch_fnames:
            fpath = filepath_lookup[fname]
            description = text_contexts.get(fname, "")
            item = _make_input_item(fpath, description)
            if item is None:
                n_failed += 1
                continue
            inputs.append(item)
            valid_fnames.append(fname)

        if not inputs:
            continue

        try:
            embeddings = _api_embed_batch(inputs, OPENROUTER_API_KEY, delay)
        except RuntimeError as e:
            logger.error(f"  Batch failed permanently: {e} — skipping {len(inputs)} images")
            n_failed += len(inputs)
            continue

        for fname, emb_vec in zip(valid_fnames, embeddings):
            emb = np.array(emb_vec, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-12:
                emb = emb / norm
            all_embs.append(emb)
            processed_filenames.append(fname)

        done_so_far = len(processed_filenames)

        # Progress log every 200 images
        if done_so_far % 200 < batch_size or batch_start + batch_size >= n:
            elapsed_batches = (batch_start // batch_size) + 1
            remaining_batches = max((n - batch_start - batch_size), 0) // batch_size
            eta_s = remaining_batches * (batch_size * delay + 0.5)
            logger.info(
                f"  [{done_so_far:,}/{total:,}] failed={n_failed}  "
                f"ETA ~{eta_s / 60:.0f} min"
            )

        # Checkpoint every CHECKPOINT_INTERVAL images
        if done_so_far % CHECKPOINT_INTERVAL < batch_size:
            _save_checkpoint(processed_filenames, np.stack(all_embs, axis=0))
            logger.info(f"  Checkpoint saved at {done_so_far:,} images")

    # ---- Save final outputs -----------------------------------------------
    embeddings_array = np.stack(all_embs, axis=0).astype(np.float32)
    logger.info(f"Saving final index — shape: {embeddings_array.shape}")

    np.save(EMBEDDINGS_PATH, embeddings_array)
    with open(FILENAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(processed_filenames, f, indent=2)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_out, f, ensure_ascii=False, indent=2)

    # Clean up checkpoint files on success
    CHECKPOINT_PATH.unlink(missing_ok=True)
    PARTIAL_NPY_PATH.unlink(missing_ok=True)

    logger.info(
        f"Done.\n"
        f"  {len(processed_filenames):,} images indexed\n"
        f"  {n_failed:,} failed\n"
        f"  {EMBEDDINGS_PATH}\n"
        f"  {FILENAMES_PATH}\n"
        f"  {METADATA_PATH}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Nemotron VL semantic image index via OpenRouter API."
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Images per API request (default 4; reduce to 1 if hitting token limits).",
    )
    parser.add_argument(
        "--delay", type=float, default=0.15,
        help="Seconds to sleep between API requests (default 0.15).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap total images processed, for smoke-testing (e.g. --limit 50).",
    )
    args = parser.parse_args()
    build_index(
        batch_size=args.batch_size,
        delay=args.delay,
        limit=args.limit,
    )
