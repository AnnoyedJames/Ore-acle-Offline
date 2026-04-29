# Retriever Evaluation - Axis: chunking
_Generated: 20260427_230020_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware | 333 | 0 | 0.451 | 0.516 | 0.442 | 0.612 | 0.265 | 0.508 | 1.000 | 0.083s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|langchain | 333 | 0 | 0.030 | 0.048 | 0.033 | 0.050 | 0.000 | 0.000 | 1.000 | 0.076s |