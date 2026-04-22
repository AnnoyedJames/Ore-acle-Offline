# Retriever Evaluation - Axis: chunking
_Generated: 20260421_230813_

| Config | N | Skipped | Recall@5 | Recall@10 | P@10 | MRR | Img Recall | PassR@10 | FPR@3 | Latency |
|--------|---|---------|----------|-----------|------|-----|------------|----------|-------|---------|
| nomic-ai/nomic-embed-text-v1.5|hybrid|section_aware | 305 | 0 | 0.070 | 0.085 | 0.053 | 0.102 | 0.027 | 0.008 | 1.000 | 0.205s |
| nomic-ai/nomic-embed-text-v1.5|hybrid|langchain | 305 | 0 | 0.032 | 0.050 | 0.035 | 0.051 | 0.000 | 0.000 | 1.000 | 0.165s |