# Retriever Evaluation - Axis: search
_Generated: 20260430_142743_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|
| nomic-ai/nomic-embed-text-v1.5|semantic|section_aware | 333 | 0 | 0.461 | 0.517 | 0.447 | 0.625 | 0.290 | 0.507 | 1.000 | 0.227s |
| nomic-ai/nomic-embed-text-v1.5|keyword_ootb|section_aware | 333 | 0 | 0.359 | 0.468 | 0.246 | 0.424 | 0.000 | 0.280 | 1.000 | 0.389s |
| nomic-ai/nomic-embed-text-v1.5|keyword|section_aware | 333 | 0 | 0.412 | 0.491 | 0.319 | 0.515 | 0.000 | 0.314 | 1.000 | 0.343s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware | 333 | 0 | 0.453 | 0.519 | 0.442 | 0.612 | 0.265 | 0.508 | 1.000 | 0.196s |