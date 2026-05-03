# Retriever Evaluation - Axis: embedding
_Generated: 20260502_032851_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|
| BAAI/bge-m3|hybrid|section_aware|zerank-2 | 333 | 0 | 0.456 | 0.530 | 0.396 | 0.599 | 0.110 | 0.472 | 1.000 | 8.470s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware|zerank-2 | 333 | 0 | 0.483 | 0.571 | 0.420 | 0.636 | 0.203 | 0.547 | 1.000 | 8.468s |
| intfloat/multilingual-e5-large|hybrid|section_aware|zerank-2 | 333 | 0 | 0.394 | 0.469 | 0.331 | 0.526 | 0.101 | 0.455 | 1.000 | 6.411s |
| baai/bge-m3|hybrid|section_aware|zerank-2 | 333 | 0 | 0.458 | 0.532 | 0.400 | 0.607 | 0.193 | 0.470 | 1.000 | 8.344s |