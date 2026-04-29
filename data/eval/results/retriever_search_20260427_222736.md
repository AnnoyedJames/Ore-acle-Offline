# Retriever Evaluation - Axis: search
_Generated: 20260427_222736_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|
| nomic-ai/nomic-embed-text-v1.5|semantic|section_aware | 333 | 0 | 0.458 | 0.514 | 0.447 | 0.625 | 0.290 | 0.507 | 1.000 | 0.302s |
| nomic-ai/nomic-embed-text-v1.5|keyword_ootb|section_aware | 333 | 0 | 0.361 | 0.467 | 0.246 | 0.425 | 0.102 | 0.000 | 1.000 | 0.120s |
| nomic-ai/nomic-embed-text-v1.5|keyword|section_aware | 333 | 0 | 0.404 | 0.481 | 0.312 | 0.513 | 0.124 | 0.000 | 1.000 | 0.066s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware | 333 | 0 | 0.450 | 0.512 | 0.440 | 0.614 | 0.268 | 0.511 | 1.000 | 0.065s |