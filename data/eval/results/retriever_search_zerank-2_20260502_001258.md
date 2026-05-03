# Retriever Evaluation - Axis: search
_Generated: 20260502_001258_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|
| nomic-ai/nomic-embed-text-v1.5|semantic|section_aware|zerank-2 | 333 | 0 | 0.479 | 0.569 | 0.419 | 0.639 | 0.204 | 0.547 | 1.000 | 8.640s |
| nomic-ai/nomic-embed-text-v1.5|keyword_ootb|section_aware|zerank-2 | 333 | 0 | 0.428 | 0.518 | 0.285 | 0.531 | 0.000 | 0.371 | 1.000 | 9.149s |
| nomic-ai/nomic-embed-text-v1.5|keyword|section_aware|zerank-2 | 333 | 0 | 0.441 | 0.543 | 0.343 | 0.587 | 0.000 | 0.395 | 1.000 | 9.017s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware|zerank-2 | 333 | 0 | 0.483 | 0.570 | 0.420 | 0.636 | 0.203 | 0.547 | 1.000 | 8.442s |