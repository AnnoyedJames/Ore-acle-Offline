# Retriever Evaluation - Axis: search
_Generated: 20260427_222638_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|
| nomic-ai/nomic-embed-text-v1.5|semantic|section_aware | 333 | 0 | 0.459 | 0.515 | 0.447 | 0.625 | 0.290 | 0.507 | 1.000 | 0.172s |
| nomic-ai/nomic-embed-text-v1.5|keyword_ootb|section_aware | 333 | 0 | 0.360 | 0.466 | 0.246 | 0.425 | 0.102 | 0.000 | 1.000 | 0.338s |
| nomic-ai/nomic-embed-text-v1.5|keyword|section_aware | 333 | 0 | 0.400 | 0.481 | 0.312 | 0.513 | 0.124 | 0.000 | 1.000 | 0.068s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware | 333 | 0 | 0.449 | 0.512 | 0.440 | 0.614 | 0.268 | 0.511 | 1.000 | 0.067s |