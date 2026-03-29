"""
Neutral Journalism Agent - Main orchestration loop.

Pipeline: Ingest → Cluster → Generate → Calibrate → Attest → Publish

Runs as a long-lived process inside an EigenCompute TEE container.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import structlog
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.attestation.onchain import create_attestation, get_tee_attestation, publish_onchain
from src.generation.calibration import calibrated_generate
from src.ingestion.fetcher import fetch_all
from src.ingestion.parser import cluster_by_topic, load_topic_settings
from src.models import SourceConfig
from src.publishing.gate import check_publish_gate
from src.publishing.publisher import publish_article, publish_thread
from src.wallet.treasury import check_funds_sufficient, estimate_cycle_cost

log = structlog.get_logger()


def load_source_settings(config_path: str = "config/sources.yaml") -> dict:
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return {
        "sources": [SourceConfig(**s) for s in data["sources"]],
        "min_sources_per_article": int(data.get("min_sources_per_article", 3)),
        "require_balanced_sourcing": bool(data.get("require_balanced_sourcing", False)),
    }


def load_sources(config_path: str = "config/sources.yaml") -> list[SourceConfig]:
    return load_source_settings(config_path)["sources"]


def _heartbeat_path() -> Path:
    return Path(os.getenv("AGENT_HEARTBEAT_PATH", "/tmp/neutral_journalism_agent.heartbeat"))


def _write_heartbeat() -> None:
    path = _heartbeat_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


async def run_cycle(
    client: AsyncOpenAI,
    generation_model: str,
    evaluator_model: str | None,
    sources: list[SourceConfig],
    topics: list[dict],
    min_source_coverage: int,
    require_balanced_sourcing: bool,
) -> int:
    """
    Run one full agent cycle. Returns the number of articles published.
    """
    # 1. Ingest
    log.info("cycle_phase", phase="ingestion")
    _write_heartbeat()
    articles = await fetch_all(sources)
    if not articles:
        log.warning("no_articles_fetched")
        return 0
    log.info("ingestion_complete", articles=len(articles))

    # 2. Cluster by topic
    log.info("cycle_phase", phase="clustering")
    clusters = cluster_by_topic(
        articles,
        topics,
        min_source_coverage=min_source_coverage,
        require_balanced_sourcing=require_balanced_sourcing,
    )
    if not clusters:
        log.warning("no_clusters_formed")
        return 0
    log.info("clustering_complete", clusters=len(clusters))

    # 3. Check funds
    max_rounds = int(os.getenv("MAX_CALIBRATION_ROUNDS", "3"))
    cost = await estimate_cycle_cost(
        num_sources=len(articles),
        max_calibration_rounds=max_rounds,
    )
    if not await check_funds_sufficient(cost * len(clusters)):
        log.error("insufficient_funds_for_cycle")
        return 0

    # 4. Generate, calibrate, attest, publish for each topic cluster
    published = 0
    tee_attestation = await get_tee_attestation()
    require_onchain_attestation = _env_flag("REQUIRE_ONCHAIN_ATTESTATION", True)

    for cluster in clusters:
        try:
            log.info("processing_cluster", topic=cluster.topic, articles=len(cluster.articles))

            # Generate + calibrate
            article = await calibrated_generate(
                cluster=cluster,
                client=client,
                generation_model=generation_model,
                evaluator_model=evaluator_model,
            )

            # Gate check before publishing
            gate = check_publish_gate(article)
            if not gate.allowed:
                log.warning(
                    "publish_blocked",
                    topic=cluster.topic,
                    reason=gate.reason,
                )
                continue

            # Attest
            attestation = await create_attestation(article, tee_attestation)
            tx_hash = await publish_onchain(attestation)
            if tx_hash:
                attestation.tx_hash = tx_hash

            gate = check_publish_gate(
                article,
                attestation=attestation,
                require_attestation=require_onchain_attestation,
            )
            if not gate.allowed:
                log.warning(
                    "publish_blocked",
                    topic=cluster.topic,
                    reason=gate.reason,
                )
                continue

            # Publish
            await publish_article(article, attestation)
            await publish_thread(article, attestation)

            published += 1
            log.info(
                "article_complete",
                topic=cluster.topic,
                headline=article.headline[:80],
                slant=article.slant_score.score if article.slant_score else None,
                rounds=article.calibration_rounds,
            )
            _write_heartbeat()

        except Exception:
            log.exception("cluster_failed", topic=cluster.topic)
            _write_heartbeat()

    return published


async def main():
    load_dotenv()
    _write_heartbeat()

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
    )

    # EigenAI client (OpenAI-compatible)
    # In production (EigenAI), auth goes via x-api-key header.
    # For local dev (OpenAI), standard api_key works too.
    eigenai_key = os.getenv("EIGENAI_API_KEY", "")
    eigenai_base = os.getenv("EIGENAI_API_BASE", "https://eigenai.eigencloud.xyz/v1")
    client = AsyncOpenAI(
        api_key=eigenai_key,
        base_url=eigenai_base,
        default_headers={"x-api-key": eigenai_key},
    )

    generation_model = os.getenv("EIGENAI_MODEL", "claude-sonnet-4-6")
    evaluator_model = os.getenv("EIGENAI_EVALUATOR_MODEL")  # None = use same model
    loop_interval = int(os.getenv("AGENT_LOOP_INTERVAL_MINUTES", "60"))

    source_settings = load_source_settings()
    topic_settings = load_topic_settings()
    sources = source_settings["sources"]
    topics = topic_settings["topics"]
    # Both configs describe the sourcing floor. Use the stricter threshold if they differ.
    min_source_coverage = max(
        source_settings["min_sources_per_article"],
        topic_settings["min_source_coverage"],
    )
    require_balanced_sourcing = source_settings["require_balanced_sourcing"]

    log.info(
        "agent_starting",
        sources=len(sources),
        topics=len(topics),
        model=generation_model,
        evaluator=evaluator_model or generation_model,
        interval_minutes=loop_interval,
        min_source_coverage=min_source_coverage,
        require_balanced_sourcing=require_balanced_sourcing,
    )

    while True:
        try:
            published = await run_cycle(
                client=client,
                generation_model=generation_model,
                evaluator_model=evaluator_model,
                sources=sources,
                topics=topics,
                min_source_coverage=min_source_coverage,
                require_balanced_sourcing=require_balanced_sourcing,
            )
            log.info("cycle_complete", published=published)
        except Exception:
            log.exception("cycle_error")
        finally:
            _write_heartbeat()

        log.info("sleeping", minutes=loop_interval)
        await asyncio.sleep(loop_interval * 60)


if __name__ == "__main__":
    asyncio.run(main())
