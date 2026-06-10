import asyncio
from backend.retrieval.answer import AnswerGenerator, GeneratorConfig

async def main():
    gen = AnswerGenerator(config=GeneratorConfig(chunking_strategy='section_aware'))
    print("\n--- Question: How do you craft an anvil? ---\n")
    # For streaming
    async for chunk in gen.generate_stream_response("How do you craft an anvil?"):
        print(chunk, end='', flush=True)
    print("\n\n--- Done ---")

if __name__ == "__main__":
    asyncio.run(main())
