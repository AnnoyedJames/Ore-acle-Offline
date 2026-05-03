# Retriever Evaluation - Axis: search
_Generated: 20260503_135655_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|
| nomic-ai/nomic-embed-text-v1.5|semantic|section_aware | 333 | 0 | 0.458 | 0.514 | 0.447 | 0.625 | 0.290 | 0.507 | 1.000 | 0.193s |
| nomic-ai/nomic-embed-text-v1.5|keyword_ootb|section_aware | 333 | 0 | 0.358 | 0.465 | 0.246 | 0.425 | 0.102 | 0.280 | 1.000 | 0.344s |
| nomic-ai/nomic-embed-text-v1.5|keyword|section_aware | 333 | 0 | 0.407 | 0.490 | 0.320 | 0.515 | 0.131 | 0.314 | 1.000 | 0.306s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware | 333 | 0 | 0.449 | 0.516 | 0.442 | 0.612 | 0.265 | 0.508 | 1.000 | 0.195s |