# Retriever Evaluation - Axis: search
_Generated: 20260427_222704_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|
| nomic-ai/nomic-embed-text-v1.5|semantic|section_aware | 333 | 0 | 0.460 | 0.515 | 0.447 | 0.625 | 0.290 | 0.507 | 1.000 | 0.367s |
| nomic-ai/nomic-embed-text-v1.5|keyword_ootb|section_aware | 333 | 0 | 0.361 | 0.468 | 0.246 | 0.425 | 0.102 | 0.000 | 1.000 | 0.119s |
| nomic-ai/nomic-embed-text-v1.5|keyword|section_aware | 333 | 0 | 0.403 | 0.481 | 0.312 | 0.513 | 0.124 | 0.000 | 1.000 | 0.065s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware | 333 | 0 | 0.452 | 0.514 | 0.440 | 0.614 | 0.268 | 0.511 | 1.000 | 0.061s |