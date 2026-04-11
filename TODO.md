 Ore-acle TODO

## Completed ✓

### Cloud → Local Migration (Feb–Mar 2026)
- [x] Git history audit (scrubbed `data/processed` to reduce repo size)
- [x] Wiki scraper — 12,487 pages
- [x] Image downloader — 61,248 WebP images
- [x] Text cleaner (HTML → structured JSON)
- [x] Section-aware chunker (512 tokens, 50-token overlap)
- [x] Embedding generator (BAAI/bge-m3 via OpenRouter API)
- [x] ChromaDB ingest — 121,080 chunks, collection `chunks_baai_bge_m3`
- [x] SQLite FTS5 ingest — 121,618 rows, `data/sqlite_fts.db`
- [x] Hybrid search (`HybridSearch` — ChromaDB + FTS5 + weighted RRF)
- [x] Frontend adapted for Local API proxy
- [x] **Fix**: SQLite FTS5 keyword queries use OR semantics (not default AND); terms ≤ 2 chars filtered as stopwords

### Evaluation Infrastructure (Apr 2026)
- [x] Two-phase ablation framework (`scripts/eval/run_eval.py`)
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

### 1. Generator Eval ← Next
Run all 4 LLMs against the winning retrieval config:
```powershell
python scripts/eval/run_eval.py --phase generator --search-mode hybrid
```
LLMs under test: Gemma 4 e2B, e4B (Ollama local), Gemma 4 31B, Gemini 3.1 Flash Lite (OpenRouter).
Metrics: Token F1, ROUGE-L, BERTScore F1, latency, token cost.
- [ ] Start Ollama (`Start-Process "ollama" -ArgumentList "serve"`) before running
- [ ] Review `data/eval/results/generator_*.md` after completion

### 2. Embedding Axis Eval
Compares BAAI/bge-m3 (done) vs three additional models. Each requires a full re-embedding + ingest pass:
- [ ] Ingest `nomic-ai/nomic-embed-text-v1.5` (768d, local) → collection `chunks_nomic_ai_nomic_embed_text_v1_5`
- [ ] Ingest `intfloat/multilingual-e5-large` (1024d, local) → collection `chunks_intfloat_multilingual_e5_large`
- [ ] Ingest `google/gemini-embedding-001` (3072d, API) → collection `chunks_google_gemini_embedding_001`
- [ ] Run `python scripts/eval/run_eval.py --phase retriever --axis embedding`

### 3. Chunking Axis Eval
Compares section-aware (done) vs LangChain recursive splitting:
- [ ] Run LangChain chunker → `data/processed/chunks_langchain.json`
- [ ] Ingest into ChromaDB collection `chunks_baai_bge_m3__langchain` + `data/sqlite_fts_langchain.db`
- [ ] Run `python scripts/eval/run_eval.py --phase retriever --axis chunking`

---

## Backlog

### FastAPI Backend
- [ ] Wire `backend/api/server.py` to `HybridSearch` (hybrid mode default)
- [ ] Serve local images via static file route
- [ ] Connect frontend proxy to FastAPI

### UI Enhancements
- [ ] Sidebar with conversation history
- [ ] Image lazy loading optimization
- [ ] Citation link persistence (URL params for shared links)

### CI / Testing
- [ ] GitHub Actions workflow
