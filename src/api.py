"""
Web app for The Neutral Wire.

Runs the autonomous pipeline in a background loop and serves the news feed UI.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import structlog
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel

from difflib import SequenceMatcher

from src.analysis.slant import evaluate_slant
from src.article_selection import select_articles_for_prompt
from src.attestation.onchain import create_attestation, get_tee_attestation, publish_onchain
from src.generation.calibration import calibrated_generate
from src.generation.writer import generate_article
from src.ingestion.fetcher import fetch_all
from src.ingestion.parser import cluster_by_topic, load_topic_settings
from src.models import GeneratedArticle, SlantScore, SourceConfig, TopicCluster
from src.publishing.gate import check_publish_gate
from src.store import ArticleStore, PipelineLog

load_dotenv()

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)
log = structlog.get_logger()

app = FastAPI(title="The Neutral Wire")
store = ArticleStore()
pipeline_log = PipelineLog()

STATIC_DIR = Path(__file__).parent.parent / "static"


# ── Helpers ────────────────────────────────────────────────

def _get_client() -> AsyncOpenAI:
    key = os.getenv("EIGENAI_API_KEY", "")
    base = os.getenv("EIGENAI_API_BASE", "https://api.openai.com/v1")
    return AsyncOpenAI(api_key=key, base_url=base, default_headers={"x-api-key": key})


def _get_model() -> str:
    return os.getenv("EIGENAI_MODEL", "gpt-4o-mini")


def _load_source_settings() -> dict:
    """Load source config - same logic as main.py daemon."""
    with open("config/sources.yaml") as f:
        data = yaml.safe_load(f)
    return {
        "sources": [SourceConfig(**s) for s in data["sources"]],
        "min_sources_per_article": int(data.get("min_sources_per_article", 3)),
        "require_balanced_sourcing": bool(data.get("require_balanced_sourcing", False)),
    }


_used_images: set[str] = set()


def _article_to_dict(article: GeneratedArticle, attestation_hash: str | None = None) -> dict:
    # Pick a unique image from sources (avoid reusing across articles)
    image_url = None
    for src in sorted(article.sources_used, key=lambda a: a.source_reliability.value, reverse=True):
        if src.image_url and src.image_url not in _used_images:
            image_url = src.image_url
            _used_images.add(image_url)
            break

    return {
        "headline": article.headline,
        "body": article.body,
        "image_url": image_url,
        "perspective": article.perspective,
        "sources": [
            {"name": a.source_name, "url": a.url, "slant": a.source_slant}
            for a in article.sources_used
        ],
        "slant": article.slant_score.model_dump() if article.slant_score else None,
        "calibration_rounds": article.calibration_rounds,
        "calibration_history": article.calibration_history,
        "model_id": article.model_id,
        "content_hash": article.content_hash,
        "prompt_hash": article.prompt_hash,
        "attestation_hash": attestation_hash,
    }


# ── Pipeline state ─────────────────────────────────────────

class PipelineState:
    def __init__(self):
        self.running = False
        self.last_run: str | None = None
        self.articles_published: int = 0
        self.cycles_completed: int = 0

pipeline_state = PipelineState()


# ── Background pipeline ───────────────────────────────────

async def run_pipeline_cycle():
    """Run one full pipeline cycle: ingest → cluster → generate → calibrate → gate → store."""
    client = _get_client()
    model = _get_model()
    evaluator = os.getenv("EIGENAI_EVALUATOR_MODEL") or model

    # Load config - same as daemon
    source_settings = _load_source_settings()
    topic_settings = load_topic_settings()
    sources = source_settings["sources"]
    topics = topic_settings["topics"]
    min_source_coverage = max(
        source_settings["min_sources_per_article"],
        topic_settings["min_source_coverage"],
    )
    require_balanced = source_settings["require_balanced_sourcing"]

    # 1. Ingest
    pipeline_log.log("ingestion", f"Fetching from {len(sources)} sources...")
    articles = await fetch_all(sources, hours_back=24)
    outlet_names = sorted({a.source_name for a in articles})
    pipeline_log.log("ingestion", f"Fetched {len(articles)} articles from {len(outlet_names)} outlets: {', '.join(outlet_names)}",
                     count=len(articles))

    if not articles:
        pipeline_log.log("ingestion", "No articles fetched, skipping cycle")
        return 0

    # 2. Cluster
    pipeline_log.log("clustering", f"Clustering by topic (min {min_source_coverage} sources, balanced={require_balanced})...")
    clusters = cluster_by_topic(
        articles, topics,
        min_source_coverage=min_source_coverage,
        require_balanced_sourcing=require_balanced,
    )
    pipeline_log.log("clustering", f"Formed {len(clusters)} topic clusters",
                     clusters=[c.topic for c in clusters])

    if not clusters:
        pipeline_log.log("clustering", "No clusters with balanced sources, skipping cycle")
        return 0

    # 3. Process each cluster
    published = 0
    pipeline_log.log("attestation", "Fetching TEE attestation...")
    tee_attestation = await get_tee_attestation()
    pipeline_log.log("attestation", f"TEE attestation ready ({len(tee_attestation)} hex chars)")

    # Deduplicate: skip topics we've already written about recently
    # Groups store headlines under "neutral"/"left"/"right" sub-dicts; legacy stores flat.
    existing_headlines = set()
    for entry in store.list_all(limit=20):
        if "neutral" in entry and isinstance(entry["neutral"], dict):
            h = entry["neutral"].get("headline", "")
        else:
            h = entry.get("headline", "")
        if h:
            existing_headlines.add(h.lower())
    pipeline_log.log("system", f"Loaded {len(existing_headlines)} existing headlines for dedupe")

    for cluster in clusters[:3]:  # Cap at 3 articles per cycle
        topic = cluster.topic

        # Keep outlet diversity and ideological balance in the prompt-sized sample.
        selected = select_articles_for_prompt(cluster.articles, max_per_source=2, max_total=8)
        by_source = {}
        for article in selected:
            by_source.setdefault(article.source_name, []).append(article)
        cluster = cluster.model_copy(update={"articles": selected})

        pipeline_log.log("generation", f"Generating article on [{topic}] from {len(selected)} articles ({len(by_source)} outlets)...",
                         topic=topic, source_count=len(selected), outlets=len(by_source))

        try:
            # Generate neutral (calibrated) + left/right (evaluated only) in parallel
            neutral_task = calibrated_generate(
                cluster=cluster,
                client=client,
                generation_model=model,
                evaluator_model=evaluator,
            )
            left_task = _generate_and_evaluate(
                cluster=cluster, client=client, model=model,
                evaluator_model=evaluator, perspective="left",
            )
            right_task = _generate_and_evaluate(
                cluster=cluster, client=client, model=model,
                evaluator_model=evaluator, perspective="right",
            )
            neutral_article, left_article, right_article = await asyncio.gather(
                neutral_task, left_task, right_task,
            )

            # Dedup: skip if headline is too similar to an existing article
            headline_lower = neutral_article.headline.lower()
            is_dup = any(
                SequenceMatcher(None, headline_lower, h).ratio() > 0.6
                for h in existing_headlines
            )
            if is_dup:
                pipeline_log.log("gate", f"[{topic}] SKIPPED: too similar to existing article", topic=topic)
                continue
            existing_headlines.add(headline_lower)

            slant = neutral_article.slant_score
            if slant:
                pipeline_log.log("evaluation", f"[{topic}] Neutral slant: {slant.overall_slant_score:+.2f}, "
                                 f"Rounds: {neutral_article.calibration_rounds}",
                                 topic=topic,
                                 slant=slant.overall_slant_score,
                                 rounds=neutral_article.calibration_rounds)

            # Gate check (neutral article only)
            gate = check_publish_gate(neutral_article)
            if not gate.allowed:
                pipeline_log.log("gate", f"[{topic}] BLOCKED: {gate.reason}", topic=topic)
                continue

            pipeline_log.log("gate", f"[{topic}] PASSED", topic=topic)

            # Attest (neutral article)
            attestation = await create_attestation(neutral_article, tee_attestation)
            tx_hash = await publish_onchain(attestation)
            if tx_hash:
                attestation.tx_hash = tx_hash

            # Store as a group: left, neutral, right
            group = {
                "topic": topic,
                "left": _article_to_dict(left_article),
                "neutral": _article_to_dict(neutral_article, attestation.article_hash),
                "right": _article_to_dict(right_article),
            }
            store.add(group)
            published += 1

            pipeline_log.log("published", f"[{topic}] Published 3 perspectives: {neutral_article.headline[:50]}...",
                             topic=topic, headline=neutral_article.headline)

        except Exception as e:
            pipeline_log.log("error", f"[{topic}] Failed: {e}", topic=topic)
            log.exception("cluster_failed", topic=topic)

    return published


async def _generate_and_evaluate(
    cluster: TopicCluster,
    client: AsyncOpenAI,
    model: str,
    evaluator_model: str,
    perspective: str,
) -> GeneratedArticle:
    """Generate a left or right article and evaluate its slant (no calibration)."""
    article = await generate_article(
        topic=cluster.topic,
        sources=cluster.articles,
        client=client,
        model=model,
        perspective=perspective,
    )
    slant = await evaluate_slant(
        headline=article.headline,
        body=article.body,
        client=client,
        evaluator_model=evaluator_model,
    )
    return article.model_copy(update={"slant_score": slant})


async def pipeline_loop():
    """Background loop that runs the pipeline on a schedule."""
    interval = int(os.getenv("AGENT_LOOP_INTERVAL_MINUTES", "60"))
    pipeline_state.running = True
    pipeline_log.log("system", f"Pipeline starting, interval={interval}m")

    while True:
        try:
            pipeline_log.log("system", "Starting pipeline cycle...")
            count = await run_pipeline_cycle()
            pipeline_state.articles_published += count
            pipeline_state.cycles_completed += 1
            from datetime import datetime, timezone
            pipeline_state.last_run = datetime.now(timezone.utc).isoformat()
            pipeline_log.log("system", f"Cycle complete: {count} articles published")
        except Exception as e:
            pipeline_log.log("error", f"Cycle failed: {e}")
            log.exception("pipeline_cycle_error")

        pipeline_log.log("system", f"Sleeping {interval} minutes...")
        await asyncio.sleep(interval * 60)


@app.on_event("startup")
async def startup():
    asyncio.create_task(pipeline_loop())


# ── API endpoints ──────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/articles")
async def api_articles(limit: int = 20):
    return store.list_all(limit=limit)


@app.get("/api/pipeline/status")
async def api_pipeline_status():
    return {
        "running": pipeline_state.running,
        "last_run": pipeline_state.last_run,
        "articles_published": pipeline_state.articles_published,
        "cycles_completed": pipeline_state.cycles_completed,
        "total_articles": store.count(),
        "model": _get_model(),
    }


@app.get("/api/pipeline/log")
async def api_pipeline_log(limit: int = 50):
    return pipeline_log.recent(limit=limit)


@app.post("/api/pipeline/trigger")
async def api_trigger():
    """Manually trigger a pipeline cycle."""
    pipeline_log.log("system", "Manual trigger received")
    asyncio.create_task(_manual_trigger())
    return {"status": "triggered"}


async def _manual_trigger():
    try:
        count = await run_pipeline_cycle()
        pipeline_state.articles_published += count
        pipeline_state.cycles_completed += 1
        from datetime import datetime, timezone
        pipeline_state.last_run = datetime.now(timezone.utc).isoformat()
        pipeline_log.log("system", f"Manual cycle complete: {count} articles")
    except Exception as e:
        pipeline_log.log("error", f"Manual cycle failed: {e}")


class EvalRequest(BaseModel):
    headline: str
    body: str


@app.post("/api/evaluate")
async def api_evaluate(req: EvalRequest):
    """Evaluate a single article (demo/debug tool)."""
    client = _get_client()
    model = _get_model()
    try:
        slant = await evaluate_slant(req.headline, req.body, client, model)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return slant.model_dump()


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": _get_model()}


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
