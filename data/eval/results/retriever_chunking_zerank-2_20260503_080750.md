# Retriever Evaluation - Axis: chunking
_Generated: 20260503_080750_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware|zerank-2 | 333 | 0 | 0.488 | 0.572 | 0.420 | 0.636 | 0.203 | 0.547 | 1.000 | 8.585s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|langchain|zerank-2 | 333 | 0 | 0.500 | 0.563 | 0.447 | 0.637 | 0.000 | 0.542 | 1.000 | 11.316s |