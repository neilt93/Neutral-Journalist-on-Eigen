"""Helpers for selecting a balanced subset of articles for prompt construction."""

from __future__ import annotations

from src.models import IngestedArticle


def _bucket_for_slant(source_slant: float) -> str:
    if source_slant < -0.15:
        return "left"
    if source_slant > 0.15:
        return "right"
    return "center"


def select_articles_for_prompt(
    articles: list[IngestedArticle],
    max_per_source: int = 2,
    max_total: int = 8,
) -> list[IngestedArticle]:
    """
    Select a prompt-sized subset without dropping entire ideological buckets.

    Strategy:
    - preserve source diversity by taking one article per source before duplicates
    - interleave left, right, and center sources so later sources are not starved
    - cap the number of articles per source and overall prompt size
    """
    if max_per_source <= 0 or max_total <= 0 or not articles:
        return []

    by_source: dict[str, list[IngestedArticle]] = {}
    for article in articles:
        by_source.setdefault(article.source_name, []).append(article)

    bucketed_sources: dict[str, list[str]] = {"left": [], "right": [], "center": []}
    for source_name, source_articles in by_source.items():
        bucketed_sources[_bucket_for_slant(source_articles[0].source_slant)].append(source_name)

    source_order: list[str] = []
    while any(bucketed_sources.values()):
        for bucket_name in ("left", "right", "center"):
            bucket = bucketed_sources[bucket_name]
            if bucket:
                source_order.append(bucket.pop(0))

    selected: list[IngestedArticle] = []
    per_source_counts = {source_name: 0 for source_name in source_order}

    while len(selected) < max_total:
        added_this_pass = False
        for source_name in source_order:
            next_index = per_source_counts[source_name]
            source_articles = by_source[source_name]
            if next_index >= len(source_articles) or next_index >= max_per_source:
                continue

            selected.append(source_articles[next_index])
            per_source_counts[source_name] += 1
            added_this_pass = True

            if len(selected) >= max_total:
                break

        if not added_this_pass:
            break

    return selected
