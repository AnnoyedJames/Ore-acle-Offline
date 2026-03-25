"""
Answer Generator — produces cited answers using DeepSeek chat API.

Takes search results from HybridSearch and generates a natural language
answer with inline [1][2] citations referencing specific chunks.

Uses the OpenAI SDK with base_url pointed at DeepSeek's API.

Usage:
    from retrieval.answer import AnswerGenerator
    gen = AnswerGenerator()
    response = gen.generate("How do I find diamonds?", search_results)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are Ore-acle, a knowledgeable and friendly Minecraft encyclopedia assistant.

You answer questions about Minecraft using ONLY the provided source chunks. Every factual claim must be supported by citing the relevant source using [n] notation, where n corresponds to the source number.

Rules:
1. Use ONLY information from the provided sources. Never fabricate facts, even if you parametrically think they are true — you must back claims up with a retrieved source or say you don't know / can't find it in the wiki.
2. Cite sources inline using [1], [2], etc. immediately after the claim they support.
3. If multiple sources support the same fact, cite all of them: [1][3].
4. If the sources don't contain enough information to answer, say so honestly.
5. Keep answers concise but thorough. Use Minecraft terminology naturally.
6. When discussing items, blocks, or mobs, mention relevant game mechanics (crafting, spawning, drops, etc.).

Formatting — your response is rendered as Markdown, so use it liberally:
- Use **bold** for item, mob, and block names on first mention.
- Use bullet lists (- item) or numbered lists (1. step) for sequences, ingredients, or multiple points.
- Use Markdown tables (| Header | ... | with |---| separator) whenever comparing stats, listing enchantment levels, crafting recipes with multiple items, mob drops, or any structured data. Tables make information much easier to scan.
- Use inline `code` formatting for commands (e.g. `/give`, `/tp`).
- Use ### subheadings to organize longer answers into logical sections.
- You may use Minecraft § color codes for emphasis (e.g., §6Gold§r for gold-colored text, §aDiamond§r for green, §c§lImportant§r for bold red). Combine codes: §l = bold, §o = italic, §r = reset.

Remember: accuracy and proper citation are more important than completeness."""


@dataclass
class GeneratorConfig:
    """Configuration for answer generation."""
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    max_tokens: int = 1024
    temperature: float = 0.3  # Low temp for factual accuracy
    # Maximum total context tokens for source chunks
    max_context_tokens: int = 3000


@dataclass
class GeneratedAnswer:
    """Response from the answer generator."""
    content: str
    citations: list[dict]  # [{id, page_title, page_url, section, cited_text}]
    images: list[dict]  # [{url, alt_text, section, caption, page_title}]
    model: str
    usage: dict  # {prompt_tokens, completion_tokens, total_tokens}


class AnswerGenerator:
    """
    Generates cited answers from search results using DeepSeek.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.client = None

    def _init_client(self):
        """Initialize OpenAI client pointed at DeepSeek."""
        if self.client is not None:
            return

        api_key = self.config.api_key
        if not api_key:
            from config.settings import settings
            api_key = settings.deepseek_api_key

        if not api_key:
            raise ValueError(
                "DeepSeek API key not configured. Set DEEPSEEK_API_KEY in .env"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
        )
        logger.info(f"DeepSeek client initialized (model: {self.config.model})")

    def _build_context(self, search_results: list) -> tuple[str, list[dict], list[dict]]:
        """
        Build the source context string and extract citations/images.

        Returns:
            (context_string, citations_list, images_list)
        """
        context_parts = []
        citations = []
        all_images = []
        total_tokens = 0

        for i, result in enumerate(search_results):
            # Respect token budget for context
            if total_tokens + result.token_count > self.config.max_context_tokens:
                logger.info(
                    f"Context budget reached at source {i + 1}/{len(search_results)} "
                    f"({total_tokens} tokens)"
                )
                break

            num = i + 1
            source_text = (
                f"[Source {num}]\n"
                f"Page: {result.page_title}\n"
                f"Section: {result.section_heading}\n"
                f"URL: {result.page_url}\n"
                f"Content:\n{result.text}\n"
            )
            context_parts.append(source_text)
            total_tokens += result.token_count

            # Build citation metadata for frontend
            citations.append({
                "id": num,
                "page_title": result.page_title,
                "page_url": result.page_url,
                "section": result.section_heading,
                "cited_text": result.text[:300],  # Truncate for UI
            })

            # Collect images from this chunk
            for img in result.images:
                img_entry = {
                    "url": img.get("url", ""),
                    "alt_text": img.get("alt_text", ""),
                    "section": img.get("section", result.section_heading),
                    "caption": img.get("caption", ""),
                    "page_title": result.page_title,
                }
                if img_entry["url"] and img_entry not in all_images:
                    all_images.append(img_entry)

        context = "\n---\n".join(context_parts)
        logger.info(
            f"Context: {len(citations)} sources, ~{total_tokens} tokens, "
            f"{len(all_images)} images"
        )
        return context, citations, all_images

    def generate(
        self,
        query: str,
        search_results: list,
        conversation_history: Optional[list[dict]] = None,
    ) -> GeneratedAnswer:
        """
        Generate a cited answer from search results.

        Args:
            query: User's question
            search_results: List of SearchResult from HybridSearch
            conversation_history: Optional previous messages for context

        Returns:
            GeneratedAnswer with content, citations, images, and usage stats
        """
        self._init_client()

        context, citations, images = self._build_context(search_results)

        # Build message list
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add conversation history (if multi-turn)
        if conversation_history:
            # Only include last few turns to stay within token limits
            for msg in conversation_history[-6:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Add the current query with sources
        user_message = (
            f"Question: {query}\n\n"
            f"Sources:\n{context}\n\n"
            f"Answer the question using the sources above. "
            f"Cite each source with [n] notation."
        )
        messages.append({"role": "user", "content": user_message})

        # Call DeepSeek
        logger.info(f"Calling DeepSeek ({self.config.model})...")
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        logger.info(
            f"Response: {len(content)} chars, "
            f"tokens: {usage['prompt_tokens']}+{usage['completion_tokens']}"
            f"={usage['total_tokens']}"
        )

        return GeneratedAnswer(
            content=content,
            citations=citations,
            images=images,
            model=self.config.model,
            usage=usage,
        )
