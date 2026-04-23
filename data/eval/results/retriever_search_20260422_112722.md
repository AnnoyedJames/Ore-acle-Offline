# Retriever Evaluation - Axis: search
_Generated: 20260422_112722_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|
| nomic-ai/nomic-embed-text-v1.5|semantic|section_aware | 305 | 0 | 0.481 | 0.536 | 0.466 | 0.648 | 0.282 | 0.507 | 1.000 | 0.060s |
| nomic-ai/nomic-embed-text-v1.5|keyword_ootb|section_aware | 305 | 0 | 0.369 | 0.475 | 0.256 | 0.440 | 0.098 | 0.000 | 1.000 | 0.083s |
| nomic-ai/nomic-embed-text-v1.5|keyword|section_aware | 305 | 0 | 0.402 | 0.486 | 0.320 | 0.503 | 0.117 | 0.000 | 1.000 | 0.057s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware | 305 | 0 | 0.464 | 0.533 | 0.462 | 0.632 | 0.264 | 0.515 | 1.000 | 0.055s |