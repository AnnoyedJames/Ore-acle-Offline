 Ore-acle TODO

## Completed ✓

### Extracted Data & Architecture Improvements
- [x] **Complete codebase cleanup**: Archived legacy scripts, Next.js caches, and evaluation binaries to `archive/` (safely `.gitignore`d).
- [x] Text cleaner (HTML → structured JSON)
- [x] **MCUI Grid parsing**: Modified `text_cleaner.py` to parse Minecraft UI CSS grids (`div.mcui`) into structured `[Crafting Recipe: ...]`, `[Recipe: ...]`, etc. text variants to ensure RAG can read crafting ingredients accurately instead of broken tables.
- [x] **Running ingestion pipeline** (`text_cleaner.py`, `chunker.py`, `ingest.py`) to propagate the new MCUI grid text into ChromaDB and SQLite. (94,382 chunks total).
- [x] **Auto-Crafting Retrieval**: Handled `[Crafting Recipe: ...]` queries via an automatic FTS5 background search injection that bypasses RRF scoring.
- [x] **Offline Constraints**: Modified `config/settings.py` adding Hugging Face offline constraints (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`) to prevent internet API calls stopping execution.
- [x] Update LLM context prompt to instruct it on how to output the extracted crafting recipes in a format the demo can parse and render (3x3 Markdown tables).
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
- [x] **Search axis eval** (305-pair questionset, nomic, section_aware):
  - Semantic: R@10=0.534
  - Hybrid:   R@10=0.542 (Winner)
  - Keyword:  R@10=0.470
- [x] **Weighted RRF** — `HybridSearch` now uses `rrf_alpha` (default 0.7) and `rrf_k=20`
  - `rrf_alpha` overridable per-instance; sweepable via `--axis rrf`

---

## Completed — Evaluation Axes

- [x] **RRF alpha sweep** — optimal α found: **0.80** (k=20)

### 1. Generator Eval 
- [x] Run all 4 LLMs against the winning retrieval config:
- [x] Review `data/eval/results/generator_*.md` after completion

### 2. Embedding Axis Eval
- [x] Ingest `nomic-ai/nomic-embed-text-v1.5` (768d, local) → `chunks_nomic_ai_nomic_embed_text_v1_5`
- [x] Ingest `intfloat/multilingual-e5-large` (1024d, local)
- [x] Ingest `google/gemini-embedding-001` (3072d, API)
- [x] Run `python scripts/eval/run_eval.py --phase retriever --axis embedding`

### 3. Chunking Axis Eval
- [x] Run LangChain chunker → `data/processed/chunks_langchain.json` (78,172 chunks, with images)
- [x] Ingest langchain into ChromaDB → `chunks_nomic_ai_nomic_embed_text_v1_5__langchain` (78,161 chunks, nomic)
- [x] Re-run section_aware chunker (improved) → 94,404 chunks; re-embed from scratch (stale cache cleared)
- [x] Ingest section_aware into ChromaDB → `chunks_nomic_ai_nomic_embed_text_v1_5` (94,382 chunks, nomic)
- [x] Rebuild SQLite FTS5 → 94,404 rows (section_aware, nomic)
- [x] Run `python scripts/eval/run_eval.py --phase retriever --axis chunking` (clean run)
  - Results:
    - section_aware defeated langchain definitively

---

## Completed — Backend Integration

### FastAPI Backend
- [x] Wire `backend/api/server.py` to `HybridSearch` (hybrid mode default).
- [x] Serve local images via static file route.
- [x] Re-architected connection parameters from old proxy configurations directly to Vite configuration.

### UI Architecture & Enhancements 
- [x] **Migrated entire application layer** from Next.js 16 to a Vite / React Single Page App architecture resolving Typescript module conflicts (`frontend/.next` and `frontend/app` removed).
- [x] **Tailwind Typography Support**: Implemented `@tailwindcss/typography` plugins so the LLM markdown table generation outputs correctly into UI components.
- [x] Sidebar with conversation history
- [x] Image lazy loading optimization
### Upcoming / Pending
- [ ] Citation link persistence (URL params for shared links)

### CI / Testing
- [x] GitHub Actions workflow / Local test suite
- [x] Run test suite to verify code functionality
