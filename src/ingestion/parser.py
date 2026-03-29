"""Cluster ingested articles by topic with quality and balance filtering."""

from __future__ import annotations

from difflib import SequenceMatcher

import structlog
import yaml

from src.models import IngestedArticle, Reliability, TopicCluster

log = structlog.get_logger()

# Numeric ranking for reliability filtering
RELIABILITY_RANK = {
    Reliability.VERY_HIGH: 4,
    Reliability.HIGH: 3,
    Reliability.MEDIUM: 2,
    Reliability.LOW: 1,
}


def load_topic_settings(config_path: str = "config/topics.yaml") -> dict:
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return {
        "topics": data["topics"],
        "min_source_coverage": int(data.get("min_source_coverage", 3)),
    }


def load_topics(config_path: str = "config/topics.yaml") -> list[dict]:
    return load_topic_settings(config_path)["topics"]


def deduplicate_articles(
    articles: list[IngestedArticle],
    similarity_threshold: float = 0.85,
) -> list[IngestedArticle]:
    """
    Remove syndicated duplicates. If two articles from different sources
    have near-identical text, keep only the one from the higher-reliability source.
    """
    if len(articles) <= 1:
        return articles

    keep: list[IngestedArticle] = []
    for article in articles:
        is_dup = False
        for kept in keep:
            # Compare first 500 chars for speed
            ratio = SequenceMatcher(
                None, article.text[:500], kept.text[:500]
            ).ratio()
            if ratio >= similarity_threshold:
                # Keep the one with higher reliability
                if RELIABILITY_RANK[article.source_reliability] > RELIABILITY_RANK[kept.source_reliability]:
                    keep.remove(kept)
                    keep.append(article)
                is_dup = True
                break
        if not is_dup:
            keep.append(article)
    return keep


def cluster_by_topic(
    articles: list[IngestedArticle],
    topics: list[dict] | None = None,
    min_source_coverage: int = 3,
    require_balanced_sourcing: bool = False,
    min_reliability: Reliability = Reliability.LOW,
) -> list[TopicCluster]:
    """
    Assign articles to topic clusters based on keyword matching.

    Filters:
    - Source reliability: articles below min_reliability are excluded
    - Deduplication: syndicated near-duplicates collapsed
    - Source coverage: minimum unique sources required
    - Balance: optionally require both left and right sources
    """
    if topics is None:
        topics = load_topics()

    # Pre-filter by reliability
    min_rank = RELIABILITY_RANK[min_reliability]
    quality_articles = [
        a for a in articles
        if RELIABILITY_RANK[a.source_reliability] >= min_rank
    ]
    if len(quality_articles) < len(articles):
        log.info(
            "articles_filtered_by_reliability",
            total=len(articles),
            kept=len(quality_articles),
            min_reliability=min_reliability.value,
        )

    # Deduplicate
    quality_articles = deduplicate_articles(quality_articles)

    clusters: dict[str, TopicCluster] = {}

    for topic_cfg in topics:
        topic_name = topic_cfg["name"]
        clusters[topic_name] = TopicCluster(
            topic=topic_name,
            articles=[],
            keywords_matched=[],
        )

    for article in quality_articles:
        text_lower = (article.title + " " + article.text[:500]).lower()
        for topic_cfg in topics:
            topic_name = topic_cfg["name"]
            keywords = [kw.lower() for kw in topic_cfg["keywords"]]
            matched = [kw for kw in keywords if kw in text_lower]
            if matched:
                clusters[topic_name].articles.append(article)
                for kw in matched:
                    if kw not in clusters[topic_name].keywords_matched:
                        clusters[topic_name].keywords_matched.append(kw)

    # Filter: only return clusters with enough source diversity
    result = []
    for cluster in clusters.values():
        unique_sources = {a.source_name for a in cluster.articles}
        if len(unique_sources) < min_source_coverage:
            continue

        if require_balanced_sourcing:
            left_sources = {a.source_name for a in cluster.articles if a.source_slant < -0.15}
            right_sources = {a.source_name for a in cluster.articles if a.source_slant > 0.15}
            if not left_sources or not right_sources:
                log.info(
                    "cluster_skipped_unbalanced_sources",
                    topic=cluster.topic,
                    unique_sources=len(unique_sources),
                    left_sources=len(left_sources),
                    right_sources=len(right_sources),
                )
                continue

        result.append(cluster)

    return sorted(result, key=lambda c: len(c.articles), reverse=True)
