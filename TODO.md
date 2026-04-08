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
- [x] Hybrid search (`HybridSearch` — ChromaDB + FTS5 + RRF)
- [x] Frontend adapted for Local API proxy
- [x] **Fix**: SQLite FTS5 keyword queries use OR semantics (not default AND); terms ≤ 2 chars filtered as stopwords

### Evaluation Infrastructure (Apr 2026)
- [x] Two-phase ablation framework (`scripts/eval/run_eval.py`)
- [x] Gold questionset generator (`scripts/eval/generate_questionset.py`)
  - Gemini 3.1 Flash Lite via OpenRouter
  - Two-pass image selection (LLM Q/A → semantic search → LLM selects 3–5 images)
  - Streams 771MB `metadata.json` via `ijson`
- [x] Gold questionset — 300 pairs (100 pages × 3 Q/A: easy/medium/hard), `data/eval/questionset.json`
- [x] `_strip_thinking()` in `run_eval.py` (handles Qwen3 `<think>` tokens)
- [x] **`relevant_images` fix** — `compute_image_recall` now extracts `local_filename` from dict entries
- [x] **Search axis eval** (300-pair questionset, baai/bge-m3, section_aware):

  | Mode | MRR | R@5 | R@10 | Img Recall | Latency |
  |------|-----|-----|------|------------|---------|
  | **Semantic** | **0.620** | 0.444 | 0.518 | **0.126** | 0.807s |
  | Keyword (OR) | 0.428 | 0.360 | 0.463 | 0.061 | **0.067s** |
  | Hybrid | 0.575 | 0.445 | 0.508 | 0.106 | 0.865s |

  **Finding**: Hybrid underperforms pure semantic. OR-expanded keyword terms introduce noisy RRF matches. **Winning config: `semantic`.**

---

## Active — Evaluation Axes

### 1. Generator Eval ← Next up
Run all 5 LLMs against the winning retrieval config (semantic, baai/bge-m3, section_aware):
```powershell
python scripts/eval/run_eval.py --phase generator --search-mode semantic
```
LLMs under test: Qwen3 0.6B, 1.7B, 4B (Ollama local), Qwen3 32B, Gemini 3.1 Flash Lite (OpenRouter).
Metrics: Token F1, ROUGE-L, latency, token cost.
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
- [ ] Wire `backend/api/server.py` to `HybridSearch` (semantic mode default)
- [ ] Serve local images via static file route
- [ ] Connect frontend proxy to FastAPI

### UI Enhancements
- [ ] Sidebar with conversation history
- [ ] Image lazy loading optimization
- [ ] Citation link persistence (URL params for shared links)

### CI / Testing
- [ ] Complete `tests/test_api.py` (real RAG pipeline)
- [ ] GitHub Actions workflow
