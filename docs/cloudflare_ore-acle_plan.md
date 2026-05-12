# Definitive Ore-acle Cloud Edition (Cloudflare Stack)

## Overview

This plan describes the definitive cloud deployment of Ore-acle on Cloudflare's free tier. The offline project (`Ore-acle-Offline`) is the research branch of the original Vercel cloud version; this plan rewrites the cloud version from scratch using all offline learnings.

- Hosting: **Cloudflare Pages + Workers**
- Semantic search: **Cloudflare Vectorize**
- Keyword search: **Cloudflare D1 with FTS5**
- Embeddings: **nomic-embed-text-v1.5** via OpenRouter
- Reranker: **0.6B OpenRouter reranker**
- Generation: **OpenRouter** with **Gemma 4** default
- Data: **Upload existing `data/processed/chunks.json`** directly

## Architecture

```text
Upload Script (local, one-time)
  chunks.json → embed (nomic-embed-text-v1.5 via OpenRouter) → Vectorize
  chunks.json → insert text → D1 FTS5 (OR semantics)

Cloudflare Runtime
  Browser → Cloudflare Pages (static React frontend)
              ↕ HTTPS
          Workers (API)
              │
    POST /api/chat
      ├─ Search Worker (Hybrid RRF α=0.80, k=20)
      │   ├─ Vectorize query (Semantic, nomic embeddings)
      │   └─ D1 FTS5 query (Keyword, OR semantics)
      ├─ Rerank Worker (0.6B via OpenRouter)
      └─ Generate Worker (OpenRouter → Gemma 4)
              │
              ▼
          Response + Citations + Images
```

## Phases

### Phase 1: Repository & Cloudflare Project Setup

1. Create a new repo named `ore-acle`, separate from `Ore-acle-Offline`.
2. Initialize a Cloudflare project with `npm create cloudflare@latest`.
3. Configure `wrangler.toml` with bindings for D1, Vectorize, and OpenRouter secrets.
4. Copy reference files from the offline workspace into a `ref/` folder.

### Phase 2: Data Seeding

5. Build `scripts/seed.ts` to:
   - Read `data/processed/chunks.json`.
   - Generate embeddings using `nomic-ai/nomic-embed-text-v1.5` via OpenRouter.
   - Upload vectors and metadata to Cloudflare Vectorize.
   - Insert text rows into Cloudflare D1.
6. Create a D1 schema with a `chunks` table and `chunks_fts` FTS5 virtual table.
7. Run the seed script and verify the expected vector and row counts.

### Phase 3: Search Pipeline

8. Implement `src/workers/search.ts` to:
   - Query Vectorize for semantic candidates.
   - Query D1 FTS5 for keyword candidates with OR semantics.
   - Merge results using weighted RRF with `α=0.80`, `k=20`.
   - Hydrate keyword-only results via Vectorize metadata.
9. Configure the Vectorize index for 768 dimensions and cosine distance.

### Phase 4: Reranker Integration

10. Implement `src/workers/rerank.ts` to call the OpenRouter 0.6B reranker endpoint and reorder candidate results.

### Phase 5: Generation Pipeline

11. Implement `src/workers/generate.ts` to call OpenRouter for generation:
    - Default model: Gemma 4
    - Fallback model: Gemini Flash Lite
    - Use `Source #N:` citation formatting to avoid bracket-mimicry issues.
12. Implement `src/workers/api/chat.ts` to chain Search → Rerank → Generate.

### Phase 6: Frontend

13. Create a React app for Cloudflare Pages.
14. Port UI components from `archive/frontend_legacy/components/`:
    - `ChatInterface`
    - `MessageBubble`
    - `SourceCard`
    - `ImageGallery`
    - `LLMSettingsPanel`
15. Port shared types and utilities from `archive/frontend_legacy/types/index.ts` and `archive/frontend_legacy/lib/minecraft-colors.ts`.
16. Apply glassmorphism styling, mobile responsiveness, and `overflow-wrap: anywhere`.
17. Configure the frontend to call the Cloudflare Workers API.

## Key Differences from the Original Cloud Version

| Aspect | Original Cloud | This Plan |
|---|---|---|
| Hosting | Vercel | Cloudflare Pages + Workers |
| Vector DB | Pinecone | Cloudflare Vectorize |
| Keyword DB | Supabase | Cloudflare D1 |
| Embeddings | e5-large | nomic-embed-text-v1.5 |
| Reranking | None | 0.6B OpenRouter reranker |
| RRF | Flat k=60 | Weighted α=0.80, k=20 |
| FTS semantics | PostgreSQL AND | SQLite FTS5 OR fix |
| Citation format | `[Source N]` | `Source #N:` |
| Default LLM | Gemini Flash Lite | Gemma 4 |
| Data source | Rescrape | existing `chunks.json` |

## Verification

1. Confirm ~121k vectors in Vectorize and ~121k rows in D1.
2. Call `/api/chat` on a known eval query and verify RRF scoring behavior.
3. Confirm the reranker reorders search candidates.
4. Confirm generation output uses `Source #N:` citations.
5. Smoke test the frontend end-to-end.

## Final Decisions

- Embeddings: `nomic-ai/nomic-embed-text-v1.5`
- Reranker: OpenRouter 0.6B
- Default generator: Gemma 4
- Search: Hybrid weighted RRF (α=0.80)
- Data: existing `chunks.json`, no rescraping
- Hosting: Cloudflare free tier
- Runtime: TypeScript throughout
