"""Fetch news articles from configured sources (RSS feeds and APIs)."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import feedparser
import httpx
import structlog
from newspaper import Article as NewspaperArticle

from src.models import IngestedArticle, SourceConfig

log = structlog.get_logger()


async def fetch_all(sources: list[SourceConfig], hours_back: int = 24) -> list[IngestedArticle]:
    """Fetch recent articles from all configured sources."""
    articles: list[IngestedArticle] = []
    async with httpx.AsyncClient(timeout=30) as client:
        for source in sources:
            try:
                if source.type == "rss":
                    batch = await _fetch_rss(source, hours_back)
                elif source.type == "api":
                    batch = await _fetch_api(client, source, hours_back)
                else:
                    log.warning("unknown_source_type", source=source.name, type=source.type)
                    continue
                articles.extend(batch)
                log.info("fetched_source", source=source.name, count=len(batch))
            except Exception:
                log.exception("fetch_failed", source=source.name)
    return articles


async def _fetch_rss(source: SourceConfig, hours_back: int) -> list[IngestedArticle]:
    """Parse an RSS feed and extract full article text."""
    loop = asyncio.get_running_loop()
    feed = await loop.run_in_executor(None, feedparser.parse, source.url)
    cutoff = _utc_now() - timedelta(hours=hours_back)

    # Filter by date first, then extract text in parallel
    candidates = []
    for entry in feed.entries[:20]:
        published = _parse_date(entry.get("published_parsed") or entry.get("updated_parsed"))
        if published and _normalize_datetime(published) < cutoff:
            continue
        candidates.append((entry, published))

    if not candidates:
        return []

    # Parallel text + image extraction
    results = await asyncio.gather(*(
        loop.run_in_executor(None, _extract_article, entry.get("link", ""))
        for entry, _ in candidates
    ))

    articles = []
    for (entry, published), (text, image_url) in zip(candidates, results):
        if not text:
            continue
        title = entry.get("title", "")
        if isinstance(title, bytes):
            title = title.decode("utf-8", errors="replace")
        articles.append(IngestedArticle(
            source_name=source.name,
            source_slant=source.slant,
            source_reliability=source.reliability,
            title=title,
            url=entry.get("link", ""),
            text=text,
            image_url=image_url,
            published_at=published,
        ))
    return articles


async def _fetch_api(client: httpx.AsyncClient, source: SourceConfig, hours_back: int) -> list[IngestedArticle]:
    """Fetch articles from a JSON API. Currently supports Guardian API format."""
    api_key = os.getenv(source.api_key_env) if source.api_key_env else None
    if source.api_key_env and not api_key:
        log.warning("missing_api_key", source=source.name, env_var=source.api_key_env)
        return []

    cutoff = _utc_now() - timedelta(hours=hours_back)
    from_date = cutoff.date().isoformat()
    params: dict = {"page-size": "20", "show-fields": "bodyText", "from-date": from_date}
    if api_key:
        params["api-key"] = api_key

    resp = await client.get(source.url, params=params)
    resp.raise_for_status()
    data = resp.json()

    articles = []
    for item in data.get("response", {}).get("results", []):
        text = item.get("fields", {}).get("bodyText", "")
        if not text:
            continue
        published_at = _parse_iso(item.get("webPublicationDate"))
        if published_at and _normalize_datetime(published_at) < cutoff:
            continue
        articles.append(IngestedArticle(
            source_name=source.name,
            source_slant=source.slant,
            source_reliability=source.reliability,
            title=item.get("webTitle", ""),
            url=item.get("webUrl", ""),
            text=text,
            published_at=published_at,
        ))
    return articles


def _extract_article(url: str) -> tuple[str, str | None]:
    """Extract full article text and top image from a URL using newspaper4k."""
    if not url:
        return "", None
    try:
        article = NewspaperArticle(url)
        article.download()
        # Guard against non-UTF-8 content from source sites
        if isinstance(article.html, bytes):
            article.html = article.html.decode("utf-8", errors="replace")
        article.parse()
        image = article.top_image if hasattr(article, "top_image") else None
        text = article.text or ""
        # Ensure text is clean UTF-8
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        return text, image or None
    except Exception:
        log.debug("extraction_failed", url=url)
        return "", None


def _parse_date(time_struct) -> datetime | None:
    if time_struct is None:
        return None
    try:
        return datetime(*time_struct[:6], tzinfo=timezone.utc)
    except Exception:
        return None


def _parse_iso(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    try:
        return _normalize_datetime(datetime.fromisoformat(date_str.replace("Z", "+00:00")))
    except Exception:
        return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
