"""Shared data models for the neutral journalism agent."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Reliability(str, Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SourceConfig(BaseModel):
    name: str
    type: str  # "rss" or "api"
    url: str
    slant: float = Field(ge=-1.0, le=1.0)
    reliability: Reliability
    api_key_env: str | None = None


class IngestedArticle(BaseModel):
    """A raw article fetched from a news source."""
    source_name: str
    source_slant: float
    source_reliability: Reliability = Reliability.MEDIUM
    title: str
    url: str
    text: str
    image_url: str | None = None
    published_at: datetime | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


class TopicCluster(BaseModel):
    """A group of articles covering the same story/topic."""
    topic: str
    articles: list[IngestedArticle]
    keywords_matched: list[str]

    @property
    def source_balance(self) -> dict[str, int]:
        """Count articles by left/center/right sourcing."""
        counts = {"left": 0, "center": 0, "right": 0}
        for a in self.articles:
            if a.source_slant < -0.15:
                counts["left"] += 1
            elif a.source_slant > 0.15:
                counts["right"] += 1
            else:
                counts["center"] += 1
        return counts


class SlantScore(BaseModel):
    """
    Structured evaluator output for slant analysis.

    Decomposes bias into distinct measurable dimensions rather than
    collapsing everything into a single score. This makes debugging,
    calibration, and attestation much cleaner.
    """
    # Bias dimensions
    overall_slant_score: float = Field(ge=-1.0, le=1.0)  # -1 left, 0 neutral, +1 right
    loaded_language_score: float = Field(ge=0.0, le=1.0)  # 0 = clean, 1 = heavily loaded
    omission_risk_score: float = Field(ge=0.0, le=1.0)  # 0 = comprehensive, 1 = major gaps
    source_balance_score: float = Field(ge=0.0, le=1.0)  # 0 = one-sided, 1 = fully balanced
    # Epistemic accuracy dimensions
    consensus_preservation_score: float = Field(ge=0.0, le=1.0)  # 0 = flattens consensus, 1 = preserves evidence weight
    attribution_score: float = Field(ge=0.0, le=1.0)  # 0 = launders claims into narrator voice, 1 = properly attributes
    # Computed / meta
    ambivalent_word_ratio: float  # Per Hall: higher = more neutral
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    pass_fail: bool  # True = within acceptable neutrality threshold

    @property
    def score(self) -> float:
        """Alias for overall_slant_score for backward compat."""
        return self.overall_slant_score


class GeneratedArticle(BaseModel):
    """An article produced by the agent."""
    headline: str
    body: str
    sources_used: list[IngestedArticle]
    slant_score: SlantScore | None = None
    calibration_rounds: int = 0
    calibration_history: list[dict] = Field(default_factory=list)
    model_id: str = ""
    prompt_hash: str = ""
    perspective: str = "neutral"  # "neutral", "left", or "right"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(
            (self.headline + self.body).encode()
        ).hexdigest()


class AttestationRecord(BaseModel):
    """Onchain attestation proving article provenance."""
    article_hash: str              # SHA-256 of headline + body
    source_set_hash: str           # SHA-256 of sorted source content hashes
    evaluator_output_hash: str     # SHA-256 of full SlantScore JSON
    prompt_config_hash: str        # SHA-256 of system prompt + user prompt
    model_id: str
    slant_score: float
    calibration_rounds: int
    tee_attestation: str           # TEE-signed proof from EigenCompute
    tx_hash: str | None = None     # Set after onchain submission
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
