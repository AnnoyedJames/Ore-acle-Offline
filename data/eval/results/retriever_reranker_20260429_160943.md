# Retriever Evaluation - Axis: reranker
_Generated: 20260429_160943_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware|no-reranker | 333 | 0 | 0.452 | 0.518 | 0.442 | 0.612 | 0.265 | 0.508 | 1.000 | 0.204s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware|BGE Reranker v2-m3 | 333 | 0 | 0.456 | 0.520 | 0.413 | 0.604 | 0.230 | 0.535 | 1.000 | 3.738s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware|Qwen3-Reranker-0.6B | 333 | 0 | 0.448 | 0.526 | 0.446 | 0.604 | 0.317 | 0.542 | 1.000 | 1.949s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware|Qwen3-Reranker-4B | 333 | 0 | 0.472 | 0.543 | 0.429 | 0.636 | 0.317 | 0.568 | 1.000 | 33.044s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware|ZeroEntropy zerank-2 | 333 | 0 | 0.483 | 0.571 | 0.420 | 0.636 | 0.203 | 0.547 | 1.000 | 13.331s |