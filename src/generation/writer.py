"""Article generation via EigenAI (OpenAI-compatible API)."""

from __future__ import annotations

import hashlib
import os

from openai import AsyncOpenAI

from src.models import GeneratedArticle, IngestedArticle

def _parse_seed() -> int | None:
    raw = os.getenv("EIGENAI_INFERENCE_SEED", "42")
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None

# Seed for deterministic inference on EigenAI.
# Same seed + same prompt = same output, with a cryptographic signature.
# Set to None to disable (non-deterministic mode for local dev).
INFERENCE_SEED = _parse_seed()

SYSTEM_PROMPT = """You are a neutral journalism agent. Your mandate is to produce factually accurate, politically balanced news articles.

Rules:
1. Present all significant perspectives on the topic without favoring any side.
2. Use precise, factual language. Avoid loaded words, emotional framing, or editorializing.
3. Attribute claims to their sources. Do not present contested claims as settled fact.
4. When experts disagree, present the range of credible positions.
5. Prioritize what happened (facts) over why it matters (interpretation).
6. Include relevant context that helps the reader understand all sides.
7. If there is genuine scientific or expert consensus, state it clearly while noting dissenting views.
8. Write in a clear, accessible style suitable for a general audience.

You will receive summaries of source articles from across the political spectrum. Synthesize them into a single balanced article."""

GENERATION_PROMPT = """Write a news article synthesizing the following source coverage. Produce a headline and body.

Topic: {topic}

Source articles:
{source_summaries}

Output format:
HEADLINE: <your headline>

<article body>"""


async def generate_article(
    topic: str,
    sources: list[IngestedArticle],
    client: AsyncOpenAI,
    model: str,
) -> GeneratedArticle:
    """Generate a neutral article from source material via EigenAI."""
    source_summaries = _format_sources(sources)
    user_prompt = GENERATION_PROMPT.format(
        topic=topic,
        source_summaries=source_summaries,
    )
    prompt_hash = hashlib.sha256(
        (SYSTEM_PROMPT + user_prompt).encode()
    ).hexdigest()

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        seed=INFERENCE_SEED,
    )

    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("Generator returned empty response")
    headline, body = _parse_output(raw)

    return GeneratedArticle(
        headline=headline,
        body=body,
        sources_used=sources,
        model_id=model,
        prompt_hash=prompt_hash,
    )


def _format_sources(sources: list[IngestedArticle]) -> str:
    parts = []
    for i, src in enumerate(sources, 1):
        # Truncate to keep prompt manageable
        excerpt = src.text[:1500]
        parts.append(
            f"[Source {i}: {src.source_name} (slant: {src.source_slant:+.2f})]\n"
            f"Title: {src.title}\n"
            f"Excerpt: {excerpt}\n"
        )
    return "\n---\n".join(parts)


def _parse_output(raw: str) -> tuple[str, str]:
    """Split model output into headline and body."""
    lines = raw.strip().split("\n", 1)
    if lines[0].upper().startswith("HEADLINE:"):
        headline = lines[0].split(":", 1)[1].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
    else:
        # Fallback: first line is headline
        headline = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
    return headline, body
