# Ore-acle TODO

## Completed (Feb 2026)
- [x] **Git History Audit**: Scrubbed `data/processed` from all commits to reduce repo size.
- [x] **Build Fixes**: Patched Pinecone SDK type errors in Next.js build.
- [x] **RAG Enhancements**: Added conversation history, context enrichment (Infoboxes), and query refinement.
- [x] **UI Polish**: Centered About page stats, improved mobile header responsiveness, tuned dark mode overlay.
- [x] **Supabase Setup**: SQL schema defines `chunks` table and fuzzy search function.
- [x] **Documentation**: Updated architecture diagrams for 1024d/Pinecone Serverless stack.
- [x] **Database Pipeline**: Ran full populate pipeline — Pinecone (347k vectors, 1024d) + Supabase (keyword index).
- [x] **Search Fix**: Fixed model mismatch in `search.py` (was Nomic 768d, now Multilingual-E5-Large 1024d).
- [x] **Semantic Tag System**: Replaced raw §-code prompting with `[heading]`, `[term]`, `[sub]`, `[tip]`, `[warning]`, `[note]` tags. LLM writes tags, renderer expands to §-codes.

## High Priority

### 1. Run Database Setup & Uploads
**Backend code is complete** — need to execute the pipeline:
- [x] Create Pinecone index (1024d, cosine, serverless) at console.pinecone.io
- [x] Run `setup.sql` in Supabase SQL Editor (creates chunks table, tsvector, RPC functions)
- [ ] Create Cloudflare R2 bucket (`ore-acle-images`) + API token
- [x] Set all env vars in `.env` (see `.env.example`)
- [x] Run `python scripts/populate_database.py` (uploads to Pinecone + Supabase)
- [ ] Run `python scripts/upload_images_r2.py` (uploads 61k WebP images to R2)

### 2. Set Vercel Environment Variables
- [ ] `PINECONE_API_KEY` — from Pinecone console (requires Inference API permission)
- [ ] `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` — from Supabase project settings
- [ ] `DEEPSEEK_API_KEY` — from DeepSeek platform
- [ ] `R2_PUBLIC_URL` — public domain for R2 bucket
- [ ] `NEXT_PUBLIC_SITE_URL` — (optional) deployment URL for API proxying

---

## Medium Priority

### 3. UI Enhancements
- [ ] Sidebar with conversation history
- [ ] Image lazy loading optimization
- [ ] Citation link persistence (URL params for shared links)
- [ ] Loading states for real API latency

### 4. Testing & CI
- [ ] Complete `tests/test_api.py` (test real RAG pipeline)
- [ ] Complete `tests/test_embeddings.py` (768d validation)
- [ ] Set up GitHub Actions workflow (`workflows/ci.yml`)
- [ ] Add integration test for Pinecone + Supabase round-trip

### 5. Documentation
- [ ] Complete `docs/ARCHITECTURE.md`
- [ ] Complete `docs/SETUP.md` (full deployment guide)
- [ ] Add Pinecone/R2/Supabase setup instructions

---

## Low Priority

### 6. Performance & Monitoring
- [ ] Add response time logging to chat route
- [ ] Cache frequent queries in Supabase
- [ ] Monitor Pinecone/HuggingFace API latency

---

## Completed ✓
- ✓ Wiki scraper (12,487 pages)
- ✓ Image downloader (61,248 images)
- ✓ Image processor (WebP compression)
- ✓ Text cleaner (HTML → JSON with structured sections)
- ✓ Chunker (recursive split, 512 tokens, 50-token overlap)
- ✓ Embedding generator (Multilingual-E5-Large, 1024d on CUDA)
- ✓ Hybrid search pipeline (Pinecone semantic + Supabase keyword + RRF)
- ✓ Answer generation (DeepSeek with inline citations)
- ✓ Dual-target uploader (Pinecone vectors+metadata + Supabase keyword index)
- ✓ R2 image upload script (boto3, resume support)
- ✓ Frontend UI (glassmorphism, dark/light themes, citations, image gallery)
- ✓ About page (architecture walkthrough, tech stack, design decisions)
- ✓ Mock chat API (for frontend testing without backend)
- ✓ Mobile responsive design (12px font, width containment chain)
- ✓ Semantic tag formatting system (LLM writes [tag]…[/tag], renderer expands to §-codes)
- ✓ Auto-send demo prompts on click
- ✓ Vercel deployment
