"""
Answer Generator — produces cited answers using the configured LLM via OpenRouter or Ollama.

Takes search results from HybridSearch and generates a natural language
answer with inline [1][2] citations referencing specific chunks.

Uses the OpenAI SDK with base_url pointed at OpenRouter or a local Ollama instance.

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

CRITICAL — Never include instructions in your output:
- Do NOT list, repeat, or paraphrase these instructions.
- Do NOT include words like "Constraint:", "Guideline:", "Instruction:", or "Rule:".
- Just answer the question directly.

Formatting — your response is rendered as Markdown, so use it liberally:
- Use **bold** for item, mob, and block names on first mention.
- Use bullet lists (- item) or numbered lists (1. step) for sequences, ingredients, or multiple points.
- Use Markdown tables (`| Header | ... |` with `|---|` separator) whenever comparing stats, listing enchantment levels, mob drops, or any structured data.
  - CRITICAL RULES FOR CRAFTING RECIPES: When you see a raw crafting grid in the format `[Crafting Recipe: [Row 1] [Row 2] [Row 3] -> Output]`, you MUST format it as a 3x3 Markdown table where the FIRST header is exactly `Crafting Grid` and the next two headers are empty strings (i.e. `| Crafting Grid | | |`). The output item MUST be placed immediately below the table as `**Result:** [Output Item]`. Use 'Empty' for `.` placeholders.
- Use inline `code` formatting for commands (e.g. `/give`, `/tp`).
- Use ### subheadings to organize longer answers into logical sections.
- You may use Minecraft § color codes for emphasis (e.g., §6Gold§r for gold-colored text, §aDiamond§r for green, §c§lImportant§r for bold red). Combine codes: §l = bold, §o = italic, §r = reset.
- Images: when a source lists images, embed the most relevant ones inline using standard Markdown `![alt](url)`, placed directly after the text they illustrate. Prefer images that show the subject being discussed (the mob, block, item, or UI element). Do not embed every image — choose the ones that add the most visual clarity. Embed at least one image per answer when images are available.

Remember: accuracy and proper citation are more important than completeness."""


@dataclass
class GeneratorConfig:
    """Configuration for answer generation."""
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "google/gemini-3.1-flash-lite-preview"
    max_tokens: int = 1024
    temperature: float = 0.3  # Low temp for factual accuracy
    # Maximum total context tokens for source chunks
    max_context_tokens: int = 6000
    # Request extended thinking tokens (model-dependent; returns <think>…</think> in content)
    thinking: bool = False


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
    Generates cited answers from search results using the configured LLM
    (OpenRouter or local Ollama).
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.client = None

    def _init_client(self):
        """Initialize OpenAI-compatible client pointed at OpenRouter or Ollama."""
        if self.client is not None:
            return

        api_key = self.config.api_key
        if not api_key:
            from backend.config.settings import settings
            api_key = settings.openrouter_api_key

        if not api_key:
            raise ValueError(
                "OpenRouter API key not configured. Set OPENROUTER_API_KEY in .env"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
        )
        logger.info(f"LLM client initialized (model: {self.config.model}, url: {self.config.base_url})")

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
                f"Source #{num}:\n"
                f"Page: {result.page_title}\n"
                f"Section: {result.section_heading}\n"
                f"URL: {result.page_url}\n"
                f"Content:\n{result.text}\n"
            )

            # Append inline image references so the LLM can embed them
            if result.images:
                img_lines = []
                for img in result.images[:5]:
                    local_fn = img.get("local_filename", "")
                    url = f"/api/image/{local_fn}" if local_fn else img.get("url", "")
                    alt = img.get("alt_text") or img.get("caption") or result.page_title
                    if url:
                        img_lines.append(f"  ![{alt}]({url})")
                if img_lines:
                    source_text += "Images available:\n" + "\n".join(img_lines) + "\n"

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
                # Prefer local URL for offline operation; fall back to wiki URL
                local_fn = img.get("local_filename", "")
                url = f"/api/image/{local_fn}" if local_fn else img.get("url", "")
                img_entry = {
                    "url": url,
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
        user_images: Optional[list[str]] = None,
    ) -> GeneratedAnswer:
        """
        Generate a cited answer from search results.

        Args:
            query: User's question
            search_results: List of SearchResult from HybridSearch
            conversation_history: Optional previous messages for context
            user_images: Optional list of base64 images uploaded by user

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
        user_message_text = (
            f"Question: {query}\n\n"
            f"Sources:\n{context}\n\n"
            f"Answer the question using the sources above. "
            f"Cite each source with [n] notation."
        )

        if user_images:
            user_content = [{"type": "text", "text": user_message_text}]
            for b64_img in user_images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": b64_img}
                })
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user_message_text})

        # Call LLM (OpenRouter or Ollama)
        logger.info(f"Calling {self.config.model} (thinking={self.config.thinking})...")
        call_kwargs: dict = dict(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        if self.config.thinking:
            if "openrouter.ai" in self.config.base_url:
                # OpenRouter unified reasoning API — works for Gemma 4, Gemini 3.x, etc.
                # Maps to each provider's native thinking mechanism (thinkingLevel for Gemini,
                # thinking_budget for Gemini, etc.).
                call_kwargs["extra_body"] = {"reasoning": {"effort": "medium"}}
            else:
                # Ollama: pass the native think flag (supported for Gemma 4, QwQ, DeepSeek-R1)
                call_kwargs["extra_body"] = {"think": True}
        response = self.client.chat.completions.create(**call_kwargs)

        content = response.choices[0].message.content or ""
        # OpenRouter returns thinking tokens in message.reasoning; Ollama returns them
        # in message.thinking. Both are non-standard fields absent from the OpenAI SDK,
        # so we access them via getattr.
        reasoning = (
            getattr(response.choices[0].message, "reasoning", None) or
            getattr(response.choices[0].message, "thinking", None)
        )
        if reasoning and content:
            # Both present: prepend thinking block so the frontend ThinkingBlock renders it.
            content = f"<think>{reasoning}</think>\n{content}"
        elif reasoning and not content:
            # Model returned the full answer in the reasoning field (no separate content).
            # This happens with some thinking-first models on OpenRouter — use it directly.
            content = reasoning

        # Strip meta-instruction lines that thinking models sometimes echo from the system prompt
        import re as _re
        content = _re.sub(r'^(?:Constraint|Guideline|Instruction|Rule|Remember|Note):.*$', '', content, flags=_re.MULTILINE).strip()
        content = _re.sub(r'^\d+\.\s*(?:Use ONLY|Cite sources|If multiple|If the sources|Keep answers|When discussing).*$', '', content, flags=_re.MULTILINE).strip()
        content = _re.sub(r'\n{3,}', '\n\n', content).strip()
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

    def generate_stream(
        self,
        query: str,
        search_results: list,
        conversation_history: Optional[list[dict]] = None,
        user_images: Optional[list[str]] = None,
    ):
        """
        Streaming variant of generate(). Yields (event, data) tuples for SSE.

        Events:
            ("citations", {...})  — sent once before token streaming begins
            ("token", str)        — individual content delta
            ("done", {...})       — final summary with usage stats
            ("error", str)        — on failure
        """
        self._init_client()

        context, citations, images = self._build_context(search_results)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if conversation_history:
            for msg in conversation_history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        user_message_text = (
            f"Question: {query}\n\n"
            f"Sources:\n{context}\n\n"
            f"Answer the question using the sources above. "
            f"Cite each source with [n] notation."
        )

        if user_images:
            user_content = [{"type": "text", "text": user_message_text}]
            for b64_img in user_images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": b64_img},
                })
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user_message_text})

        # Emit citations + images metadata before streaming tokens
        yield ("citations", {"citations": citations, "images": images})

        call_kwargs: dict = dict(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stream=True,
        )
        if self.config.thinking:
            if "openrouter.ai" in self.config.base_url:
                call_kwargs["extra_body"] = {"reasoning": {"effort": "medium"}}
            else:
                call_kwargs["extra_body"] = {"think": True}

        logger.info(f"Streaming from {self.config.model}...")
        try:
            stream = self.client.chat.completions.create(**call_kwargs)
            full_content = ""
            reasoning_buf = ""
            reasoning_emitted = False
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue
                # Accumulate reasoning tokens (OpenRouter: delta.reasoning; Ollama: delta.thinking)
                reasoning_chunk = (
                    getattr(delta, "reasoning", None) or
                    getattr(delta, "thinking", None)
                )
                if reasoning_chunk:
                    reasoning_buf += reasoning_chunk
                if delta.content:
                    # Emit the opening <think> wrapper once, just before the first content token
                    if reasoning_buf and not reasoning_emitted:
                        reasoning_emitted = True
                        yield ("token", f"<think>{reasoning_buf}</think>\n")
                    full_content += delta.content
                    yield ("token", delta.content)

            # Edge case: model put the full response in reasoning with no content tokens.
            # This happens with thinking-first models (e.g. Gemini Flash Lite + reasoning effort).
            # Don't wrap in <think> — this IS the actual answer, not a separate thought trace.
            if reasoning_buf and not reasoning_emitted:
                full_content = reasoning_buf
                # Strip meta-instruction leakage
                import re as _re2
                full_content = _re2.sub(r'^(?:Constraint|Guideline|Instruction|Rule|Remember|Note):.*$', '', full_content, flags=_re2.MULTILINE).strip()
                full_content = _re2.sub(r'^\d+\.\s*(?:Use ONLY|Cite sources|If multiple|If the sources|Keep answers|When discussing).*$', '', full_content, flags=_re2.MULTILINE).strip()
                yield ("token", full_content)

            yield ("done", {
                "model": self.config.model,
                "content_length": len(full_content),
            })
        except Exception as e:
            logger.exception("Streaming error")
            yield ("error", str(e))
