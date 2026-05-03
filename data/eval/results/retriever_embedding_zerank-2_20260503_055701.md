# Retriever Evaluation - Axis: embedding
_Generated: 20260503_055701_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware|zerank-2 | 333 | 0 | 0.483 | 0.568 | 0.420 | 0.636 | 0.203 | 0.547 | 1.000 | 8.704s |
| intfloat/multilingual-e5-large|hybrid|section_aware|zerank-2 | 333 | 0 | 0.395 | 0.470 | 0.331 | 0.526 | 0.101 | 0.455 | 1.000 | 6.453s |
| baai/bge-m3|hybrid|section_aware|zerank-2 | 333 | 0 | 0.456 | 0.531 | 0.400 | 0.607 | 0.193 | 0.470 | 1.000 | 8.392s |