"""Helpers for selecting representative media for generated stories."""

from __future__ import annotations

import re

from src.models import GeneratedArticle, Reliability

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "were", "will", "with",
}
_RELIABILITY_SCORE = {
    Reliability.VERY_HIGH: 4,
    Reliability.HIGH: 3,
    Reliability.MEDIUM: 2,
    Reliability.LOW: 1,
}


def _keyword_tokens(text: str, limit: int = 120) -> set[str]:
    tokens = [
        token for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    ]
    return set(tokens[:limit])


def pick_representative_image(article: GeneratedArticle) -> str | None:
    """Choose the most relevant source image for a generated article."""
    article_title_tokens = _keyword_tokens(article.headline, limit=24)
    article_body_tokens = _keyword_tokens(article.body, limit=160)

    best_url = None
    best_score = float("-inf")

    for index, source in enumerate(article.sources_used):
        if not source.image_url:
            continue

        source_title_tokens = _keyword_tokens(source.title, limit=24)
        source_text_tokens = _keyword_tokens(source.text, limit=120)

        headline_overlap = len(article_title_tokens & source_title_tokens)
        contextual_overlap = len(
            (article_title_tokens | article_body_tokens) & (source_title_tokens | source_text_tokens)
        )
        reliability_bonus = _RELIABILITY_SCORE.get(source.source_reliability, 2)

        score = (headline_overlap * 12) + (contextual_overlap * 3) + reliability_bonus - (index * 0.1)
        if score > best_score:
            best_score = score
            best_url = source.image_url

    return best_url
