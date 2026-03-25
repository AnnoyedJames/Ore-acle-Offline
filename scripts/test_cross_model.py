"""Test cross-model search: llama-text-embed-v2 query vs Nomic v1.5 index."""
import sys
sys.path.insert(0, "backend")
from config.settings import settings
from pinecone import Pinecone

pc = Pinecone(api_key=settings.pinecone_api_key)

# Embed query with llama-text-embed-v2
emb = pc.inference.embed(
    model="llama-text-embed-v2",
    inputs=[{"text": "search_query: how to find diamonds in minecraft"}],
    parameters={"input_type": "query"},
)
vec = emb.data[0].values
print(f"Query embedding dim: {len(vec)}")

# Search our Nomic-embedded index
idx = pc.Index("ore-acle")
r = idx.query(vector=vec, top_k=10, include_metadata=True)

print("\nResults (llama-text-embed-v2 query vs Nomic v1.5 index):")
print("-" * 80)
for m in r.matches:
    meta = m.metadata or {}
    title = meta.get("page_title", "?")
    section = meta.get("section_heading", "?")
    text_preview = meta.get("text", "")[:100]
    print(f"  {m.score:.4f}  {title} > {section}")
    print(f"           {text_preview}...")
    print()
