# Retriever Evaluation - Axis: search
_Generated: 20260427_215949_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|
| nomic-ai/nomic-embed-text-v1.5|semantic|section_aware | 333 | 0 | 0.460 | 0.517 | 0.447 | 0.625 | 0.290 | 0.507 | 1.000 | 0.077s |
| nomic-ai/nomic-embed-text-v1.5|keyword_ootb|section_aware | 333 | 0 | 0.361 | 0.470 | 0.246 | 0.425 | 0.102 | 0.000 | 1.000 | 0.117s |
| nomic-ai/nomic-embed-text-v1.5|keyword|section_aware | 333 | 0 | 0.397 | 0.482 | 0.310 | 0.499 | 0.116 | 0.000 | 1.000 | 0.068s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware | 333 | 0 | 0.451 | 0.516 | 0.443 | 0.612 | 0.271 | 0.514 | 1.000 | 0.067s |