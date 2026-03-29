"""
Publishing via AgentKit.

AgentKit gives the agent economic autonomy: it can publish to social media,
manage its own treasury, and pay for compute. This module handles the
publishing side.
"""

from __future__ import annotations

import os

import httpx
import structlog

from src.models import AttestationRecord, GeneratedArticle

log = structlog.get_logger()


async def publish_article(
    article: GeneratedArticle,
    attestation: AttestationRecord,
) -> dict:
    """
    Publish a generated article via AgentKit.
    Returns publication metadata (URLs, IDs).
    """
    api_base = os.getenv("AGENTKIT_API_BASE", "https://agentkit.eigen.ai")
    api_key = os.getenv("AGENTKIT_API_KEY")

    if not api_key:
        log.warning("agentkit_not_configured")
        return {"status": "skipped", "reason": "AGENTKIT_API_KEY not set"}

    payload = {
        "type": "article",
        "headline": article.headline,
        "body": article.body,
        "metadata": {
            "sources": [
                {"name": a.source_name, "url": a.url}
                for a in article.sources_used
            ],
            "slant_score": article.slant_score.overall_slant_score if article.slant_score else None,
            "calibration_rounds": article.calibration_rounds,
            "model_id": article.model_id,
            "attestation_tx": attestation.tx_hash,
            "article_hash": attestation.article_hash,
        },
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{api_base}/v1/publish",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        result = resp.json()

    log.info("article_published", publication_id=result.get("id"))
    return result


async def publish_thread(
    article: GeneratedArticle,
    attestation: AttestationRecord,
) -> dict:
    """
    Publish a summary thread to social media (e.g., Twitter/X) via AgentKit.
    """
    api_base = os.getenv("AGENTKIT_API_BASE", "https://agentkit.eigen.ai")
    api_key = os.getenv("AGENTKIT_API_KEY")

    if not api_key:
        return {"status": "skipped", "reason": "AGENTKIT_API_KEY not set"}

    # Build a concise thread from the article
    thread = _article_to_thread(article, attestation)

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{api_base}/v1/social/thread",
            json={"posts": thread},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        return resp.json()


MAX_POST_LENGTH = 280


def _truncate(text: str, limit: int = MAX_POST_LENGTH) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _article_to_thread(
    article: GeneratedArticle,
    attestation: AttestationRecord,
) -> list[str]:
    """Convert an article into a social media thread. All posts capped at 280 chars."""
    thread = []

    # Post 1: Headline + hook
    thread.append(_truncate(
        f"{article.headline}\n\n"
        f"A thread on what's happening, sourced from {len(article.sources_used)} outlets "
        f"across the political spectrum."
    ))

    # Post 2-N: Key paragraphs
    paragraphs = article.body.split("\n\n")
    for para in paragraphs[:4]:
        thread.append(_truncate(para))

    # Final post: Verifiability
    slant_str = f"{article.slant_score.overall_slant_score:+.2f}" if article.slant_score else "N/A"
    thread.append(_truncate(
        f"Cryptographically verifiable.\n"
        f"Slant: {slant_str} | Sources: {len(article.sources_used)} | "
        f"Rounds: {article.calibration_rounds} | "
        f"Tx: {attestation.tx_hash or 'pending'}"
    ))

    return thread
