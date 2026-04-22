# Retriever Evaluation - Axis: search
_Generated: 20260422_000651_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|
| nomic-ai/nomic-embed-text-v1.5|semantic|section_aware | 305 | 0 | 0.064 | 0.081 | 0.050 | 0.073 | 0.024 | 0.007 | 1.000 | 0.119s |
| nomic-ai/nomic-embed-text-v1.5|keyword|section_aware | 305 | 0 | 0.364 | 0.468 | 0.256 | 0.440 | 0.098 | 0.000 | 1.000 | 0.122s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware | 305 | 0 | 0.070 | 0.085 | 0.053 | 0.102 | 0.027 | 0.008 | 1.000 | 0.156s |