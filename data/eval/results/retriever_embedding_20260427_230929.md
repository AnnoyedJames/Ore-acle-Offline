# Retriever Evaluation - Axis: embedding
_Generated: 20260427_230929_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|
| BAAI/bge-m3|hybrid|section_aware | 333 | 0 | 0.450 | 0.508 | 0.415 | 0.595 | 0.123 | 0.000 | 1.000 | 0.270s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware | 333 | 0 | 0.455 | 0.519 | 0.442 | 0.612 | 0.265 | 0.508 | 1.000 | 0.081s |
| intfloat/multilingual-e5-large|hybrid|section_aware | 333 | 0 | 0.371 | 0.432 | 0.344 | 0.508 | 0.141 | 0.403 | 1.000 | 0.094s |
| baai/bge-m3|hybrid|section_aware | 333 | 0 | 0.454 | 0.512 | 0.422 | 0.605 | 0.225 | 0.000 | 1.000 | 0.246s |