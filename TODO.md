 Ore-acle TODO

## Completed ✓

### Extracted Data Improvements
- [x] Text cleaner (HTML → structured JSON)
- [x] **MCUI Grid parsing**: Modified `text_cleaner.py` to parse Minecraft UI CSS grids (`div.mcui`) into structured `[Crafting Recipe: ...]`, `[Recipe: ...]`, etc. text variants to ensure RAG can read crafting ingredients accurately instead of broken tables.
- [ ] Run ingestion pipeline (`text_cleaner.py`, `chunker.py`, `ingest.py`) to propagate the new MCUI grid text into ChromaDB and SQLite.
- [ ] Update frontend/demo to visually render `[Crafting Recipe: ...]` grids.
- [x] Update LLM context prompt to instruct it on how to output the extracted crafting recipes in a format the demo can parse and render.
- [x] Section-aware chunker (512 tokens, 50-token overlap)
- [x] Section-aware chunker **improvements** — merge_threshold 100, absorb-backward rule, skip navigation sections → **94,404 chunks** (down from 121,618)
- [x] Embedding generator (BAAI/bge-m3 via OpenRouter API)
- [x] Default embedding model changed to `nomic-ai/nomic-embed-text-v1.5` (local, 768d)
- [x] ChromaDB ingest — `chunks_nomic_ai_nomic_embed_text_v1_5` (94,382 chunks, nomic, section_aware)
- [x] ChromaDB ingest — `chunks_nomic_ai_nomic_embed_text_v1_5__langchain` (78,161 chunks, nomic, langchain)
- [x] SQLite FTS5 ingest — 94,404 rows (section_aware), `data/sqlite_fts.db`
- [x] Hybrid search (`HybridSearch` — ChromaDB + FTS5 + weighted RRF)
- [x] Frontend adapted for Local API proxy
- [x] **Fix**: SQLite FTS5 keyword queries use OR semantics (not default AND); terms ≤ 2 chars filtered as stopwords

### Evaluation Infrastructure (Apr 2026)
- [x] Two-phase ablation framework (`scripts/eval/run_eval.py`)
- [x] Added `scripts/langchain_chunker.py` (Outputs separate `chunks_langchain.json` to not overwrite original)
- [x] Gold questionset generator (`scripts/eval/generate_questionset.py`)
- [x] Gold questionset — 305 pairs at `data/eval/questionset.json`
- [x] `_strip_thinking()` in `run_eval.py` (handles Gemma 4 `<think>` tokens)
- [x] **`relevant_images` fix** — `compute_image_recall` now extracts `local_filename` from dict entries
- [x] **Search axis eval** (305-pair questionset, baai/bge-m3, section_aware):
  - Semantic: MRR=0.620, R@5=0.444, R@10=0.518
  - Hybrid:   MRR=0.575, R@5=0.445, R@10=0.508
  - Keyword:  MRR=0.428, R@5=0.360, R@10=0.463
- [x] **Weighted RRF** — `HybridSearch` now uses `rrf_alpha` (default 0.7) and `rrf_k=20`
  - `rrf_alpha` overridable per-instance; sweepable via `--axis rrf`

---

## Active — Evaluation Axes

- [x] **RRF alpha sweep** — optimal α found: **0.80** (k=20)
  - Results (`data/eval/results/retriever_rrf_20260410_121409.md`):
    - Semantic baseline: MRR=0.620, R@5=0.446, R@10=0.519, P@10=0.424, ImgRecall=0.123
    - **Hybrid α=0.80**: MRR=0.604, R@5=0.448, R@10=0.520, P@10=0.427, ImgRecall=0.131 ← wins R@10, P@10, ImgRecall
    - Hybrid beats semantic on all coverage metrics; MRR gap is -1.6pp (top-1 precision)
  - `rrf_alpha` default updated to `0.80` in `settings.py`

### 1. Generator Eval 
- [x] Run all 4 LLMs against the winning retrieval config:
- [x] Review `data/eval/results/generator_*.md` after completion

### 2. Embedding Axis Eval
- [x] Ingest `nomic-ai/nomic-embed-text-v1.5` (768d, local) → `chunks_nomic_ai_nomic_embed_text_v1_5`
- [x] Ingest `intfloat/multilingual-e5-large` (1024d, local)
- [x] Ingest `google/gemini-embedding-001` (3072d, API)
- [x] Run `python scripts/eval/run_eval.py --phase retriever --axis embedding`
  - Results (`data/eval/results/retriever_embedding_20260415_234403.md`):
    - nomic-ai/nomic-embed-text-v1.5: MRR=0.607, R@5=0.462, R@10=0.538, ImgRecall=0.122 ← best MRR & coverage
    - BAAI/bge-m3: MRR=0.591, R@5=0.448, R@10=0.513, ImgRecall=0.126
    - intfloat/multilingual-e5-large: MRR=0.497, R@5=0.395, R@10=0.449
    - google/gemini-embedding-001: MRR=0.429, R@5=0.360, R@10=0.464 (lowest)
  - **Winner**: nomic-ai/nomic-embed-text-v1.5 on all text metrics; bge-m3 wins ImgRecall by 0.4pp

### 3. Chunking Axis Eval
- [x] Run LangChain chunker → `data/processed/chunks_langchain.json` (78,172 chunks, with images)
- [x] Ingest langchain into ChromaDB → `chunks_nomic_ai_nomic_embed_text_v1_5__langchain` (78,161 chunks, nomic)
- [x] Re-run section_aware chunker (improved) → 94,404 chunks; re-embed from scratch (stale cache cleared)
- [x] Ingest section_aware into ChromaDB → `chunks_nomic_ai_nomic_embed_text_v1_5` (94,382 chunks, nomic)
- [x] Rebuild SQLite FTS5 → 94,404 rows (section_aware, nomic)
- [ ] Run `python -m scripts.eval.run_eval --phase retriever --axis chunking` (clean run)

---

## Active — Backend Integration

### FastAPI Backend
- [ ] Wire `backend/api/server.py` to `HybridSearch` (hybrid mode default)
- [ ] Serve local images via static file route
- [ ] Connect frontend proxy to FastAPI

### UI Enhancements
- [x] Sidebar with conversation history
- [x] Image lazy loading optimization
- [ ] Citation link persistence (URL params for shared links)

### CI / Testing
- [x] GitHub Actions workflow / Local test suite
- [x] Run test suite to verify code functionality
