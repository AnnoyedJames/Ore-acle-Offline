# Retriever Evaluation - Axis: chunking
_Generated: 20260502_062453_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware|zerank-2 | 333 | 0 | 0.486 | 0.573 | 0.420 | 0.636 | 0.203 | 0.547 | 1.000 | 8.557s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|langchain|zerank-2 | 333 | 0 | 0.044 | 0.054 | 0.038 | 0.084 | 0.000 | 0.000 | 1.000 | 12.140s |