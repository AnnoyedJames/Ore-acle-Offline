# Retriever Evaluation - Axis: search
_Generated: 20260422_093456_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|
| nomic-ai/nomic-embed-text-v1.5|semantic|section_aware | 305 | 0 | 0.478 | 0.533 | 0.466 | 0.648 | 0.282 | 0.507 | 1.000 | 0.089s |
| nomic-ai/nomic-embed-text-v1.5|keyword|section_aware | 305 | 0 | 0.366 | 0.470 | 0.256 | 0.440 | 0.098 | 0.000 | 1.000 | 0.113s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware | 305 | 0 | 0.474 | 0.534 | 0.454 | 0.617 | 0.243 | 0.519 | 1.000 | 0.154s |