# Retriever Evaluation - Axis: search
_Generated: 20260503_132735_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|
| nomic-ai/nomic-embed-text-v1.5|semantic|section_aware | 333 | 0 | 0.459 | 0.515 | 0.447 | 0.625 | 0.290 | 0.507 | 1.000 | 0.287s |
| nomic-ai/nomic-embed-text-v1.5|keyword_ootb|section_aware | 333 | 0 | 0.359 | 0.466 | 0.246 | 0.424 | 0.000 | 0.280 | 1.000 | 0.363s |
| nomic-ai/nomic-embed-text-v1.5|keyword|section_aware | 333 | 0 | 0.410 | 0.491 | 0.319 | 0.515 | 0.000 | 0.314 | 1.000 | 0.285s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware | 333 | 0 | 0.452 | 0.518 | 0.442 | 0.612 | 0.265 | 0.508 | 1.000 | 0.193s |