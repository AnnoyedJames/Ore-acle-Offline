# Retriever Evaluation - Axis: chunking
_Generated: 20260419_223312_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|---------|
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware | 305 | 0 | 0.030 | 0.046 | 0.028 | 0.044 | 0.001 | 0.100s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|langchain | 305 | 0 | 0.475 | 0.544 | 0.494 | 0.658 | 0.393 | 0.075s |