"""
Tests for the slant evaluation pipeline.

Three layers:
1. Unit tests — deterministic logic (ambivalent ratio, source balance, thresholds)
2. Behavioral tests — gold set articles with expected evaluator outcomes
3. Regression tests — specific subtle failure modes

Unit tests run without any LLM. Behavioral and regression tests require
an LLM evaluator (set EIGENAI_API_KEY and EIGENAI_API_BASE, or they skip).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from dotenv import dotenv_values

if "feedparser" not in sys.modules:
    feedparser_stub = types.ModuleType("feedparser")
    feedparser_stub.parse = MagicMock()
    sys.modules["feedparser"] = feedparser_stub

if "newspaper" not in sys.modules:
    newspaper_stub = types.ModuleType("newspaper")

    class _StubArticle:
        def __init__(self, *_args, **_kwargs):
            self.text = ""

        def download(self):
            return None

        def parse(self):
            return None

    newspaper_stub.Article = _StubArticle
    sys.modules["newspaper"] = newspaper_stub

from src.analysis.slant import (
    AMBIVALENT_MARKERS,
    compute_ambivalent_ratio,
    evaluate_slant,
    evaluate_source_balance,
)
from src.ingestion.fetcher import _fetch_api
from src.ingestion.parser import cluster_by_topic
from src.models import IngestedArticle, Reliability, SlantScore, SourceConfig, TopicCluster
from tests.gold_set import GOLD_SET, GoldCase


# ============================================================
# UNIT TESTS — deterministic, no LLM needed
# ============================================================

class TestAmbivalentWordRatio:
    """Test the ambivalent/hedging word ratio computation."""

    def test_empty_text(self):
        assert compute_ambivalent_ratio("") == 0.0

    def test_no_ambivalent_words(self):
        text = "The bill passed. Lawmakers voted. The president signed it."
        assert compute_ambivalent_ratio(text) == 0.0

    def test_single_marker(self):
        text = "The bill passed however some disagreed with it entirely."
        ratio = compute_ambivalent_ratio(text)
        assert ratio > 0.0
        # "however" is 1 match in 9 words
        assert abs(ratio - 1 / 9) < 0.01

    def test_multiple_markers(self):
        text = (
            "However the bill passed. On the other hand critics say "
            "it may fail. Although proponents disagree."
        )
        ratio = compute_ambivalent_ratio(text)
        # Should find: "however", "on the other hand", "critics say", "although", "proponents"
        assert ratio > 0.0

    def test_high_ambivalence_text(self):
        """Text saturated with hedging should have a high ratio."""
        text = (
            "However on the other hand some argue that critics say "
            "while some others believe the issue is nuanced and complex. "
            "Proponents and opponents have varying perspectives."
        )
        ratio = compute_ambivalent_ratio(text)
        assert ratio > 0.15  # Very hedged text

    def test_case_insensitive(self):
        text = "HOWEVER the debate CONTINUES. Critics Say otherwise."
        ratio = compute_ambivalent_ratio(text)
        assert ratio > 0.0

    def test_markers_must_be_whole_words(self):
        """'however' inside 'whatsoeverhowever' should not match."""
        # Our regex uses \b word boundaries
        text = "The compound word whatsoeverhowever appeared."
        ratio = compute_ambivalent_ratio(text)
        assert ratio == 0.0

    def test_multi_word_markers(self):
        """Multi-word markers like 'on the other hand' should match."""
        text = "The policy is popular. On the other hand it is expensive."
        ratio = compute_ambivalent_ratio(text)
        assert ratio > 0.0


class TestSourceBalance:
    """Test source balance evaluation."""

    def test_empty_sources(self):
        result = evaluate_source_balance([])
        assert result["balanced"] is False
        assert result["reason"] == "no sources"

    def test_all_left_sources(self):
        result = evaluate_source_balance([-0.4, -0.3, -0.5])
        assert result["balanced"] is False
        assert result["has_left"] is True
        assert result["has_right"] is False

    def test_all_right_sources(self):
        result = evaluate_source_balance([0.3, 0.5, 0.4])
        assert result["balanced"] is False
        assert result["has_left"] is False
        assert result["has_right"] is True

    def test_balanced_sources(self):
        result = evaluate_source_balance([-0.4, 0.0, 0.3])
        assert result["balanced"] is True
        assert result["has_left"] is True
        assert result["has_right"] is True
        assert result["has_center"] is True

    def test_balanced_without_center(self):
        """Left + right without center should still be balanced."""
        result = evaluate_source_balance([-0.4, 0.4])
        assert result["balanced"] is True
        assert result["has_center"] is False

    def test_boundary_values(self):
        """Sources at exactly ±0.15 are center."""
        result = evaluate_source_balance([-0.15, 0.15])
        assert result["balanced"] is False  # Both are center
        assert result["has_center"] is True
        assert result["has_left"] is False
        assert result["has_right"] is False

    def test_just_outside_boundary(self):
        result = evaluate_source_balance([-0.16, 0.16])
        assert result["balanced"] is True

    def test_average_slant_computation(self):
        result = evaluate_source_balance([-0.4, 0.0, 0.4])
        assert result["avg_source_slant"] == 0.0

    def test_skewed_average(self):
        result = evaluate_source_balance([-0.5, -0.3, 0.2])
        assert result["avg_source_slant"] < 0  # Leans left


class TestSlantScoreModel:
    """Test the SlantScore Pydantic model."""

    def test_pass_when_within_threshold(self):
        score = SlantScore(
            overall_slant_score=0.10,
            loaded_language_score=0.2,
            omission_risk_score=0.1,
            source_balance_score=0.8,
            consensus_preservation_score=0.8,
            attribution_score=0.9,
            ambivalent_word_ratio=0.05,
            confidence=0.9,
            rationale="Slightly right-leaning but within threshold",
            pass_fail=True,
        )
        assert score.pass_fail is True
        assert score.score == 0.10  # backward compat alias

    def test_fail_when_outside_threshold(self):
        score = SlantScore(
            overall_slant_score=-0.45,
            loaded_language_score=0.7,
            omission_risk_score=0.6,
            source_balance_score=0.2,
            consensus_preservation_score=0.5,
            attribution_score=0.5,
            ambivalent_word_ratio=0.01,
            confidence=0.85,
            rationale="Strong left-leaning bias",
            pass_fail=False,
        )
        assert score.pass_fail is False
        assert score.score == -0.45

    def test_boundary_pass(self):
        score = SlantScore(
            overall_slant_score=0.15,
            loaded_language_score=0.1,
            omission_risk_score=0.1,
            source_balance_score=0.9,
            consensus_preservation_score=0.8,
            attribution_score=0.9,
            ambivalent_word_ratio=0.04,
            confidence=0.8,
            rationale="At threshold boundary",
            pass_fail=True,  # 0.15 == threshold, should pass
        )
        assert score.pass_fail is True

    def test_validation_rejects_out_of_range(self):
        with pytest.raises(Exception):
            SlantScore(
                overall_slant_score=1.5,  # Out of range
                loaded_language_score=0.2,
                omission_risk_score=0.1,
                source_balance_score=0.8,
                consensus_preservation_score=0.5,
                attribution_score=0.5,
                ambivalent_word_ratio=0.05,
                confidence=0.9,
                rationale="test",
                pass_fail=False,
            )


class TestTopicClusterBalance:
    """Test the TopicCluster.source_balance property."""

    def _make_article(self, source_slant: float) -> IngestedArticle:
        return IngestedArticle(
            source_name=f"Source_{source_slant}",
            source_slant=source_slant,
            title="Test",
            url="https://example.com",
            text="Test article body.",
        )

    def test_balanced_cluster(self):
        cluster = TopicCluster(
            topic="test",
            articles=[
                self._make_article(-0.4),
                self._make_article(0.0),
                self._make_article(0.3),
            ],
            keywords_matched=["test"],
        )
        balance = cluster.source_balance
        assert balance["left"] == 1
        assert balance["center"] == 1
        assert balance["right"] == 1

    def test_all_left_cluster(self):
        cluster = TopicCluster(
            topic="test",
            articles=[
                self._make_article(-0.4),
                self._make_article(-0.3),
            ],
            keywords_matched=["test"],
        )
        balance = cluster.source_balance
        assert balance["left"] == 2
        assert balance["center"] == 0
        assert balance["right"] == 0


class TestClusterByTopicFiltering:
    """Test topic clustering policy filters."""

    _TEXTS = {
        "NPR": "The economy showed mixed signals this quarter as unemployment fell to 3.7% while wage growth remained stagnant according to the Bureau of Labor Statistics.",
        "Guardian": "Economic analysts warn that rising inequality threatens long-term growth prospects with the top 1% capturing most gains from recent jobs expansion.",
        "Reuters": "Global economic indicators point to a moderate slowdown in growth for major economies with trade tensions creating uncertainty for businesses.",
        "Fox": "The booming economy added 300,000 jobs last month crushing expectations and vindicating supply-side economic policies championed by conservatives.",
    }

    def _make_article(self, source_name: str, source_slant: float) -> IngestedArticle:
        text = self._TEXTS.get(source_name, f"Unique economy coverage from {source_name}.")
        return IngestedArticle(
            source_name=source_name,
            source_slant=source_slant,
            title="Economy update",
            url=f"https://example.com/{source_name}",
            text=text,
        )

    def test_requires_balanced_sourcing_when_enabled(self):
        articles = [
            self._make_article("NPR", -0.35),
            self._make_article("Guardian", -0.40),
            self._make_article("Reuters", -0.05),
        ]

        clusters = cluster_by_topic(
            articles,
            topics=[{"name": "economy", "keywords": ["economy"]}],
            min_source_coverage=3,
            require_balanced_sourcing=True,
        )

        assert clusters == []

    def test_balanced_cluster_still_passes_policy_filter(self):
        articles = [
            self._make_article("NPR", -0.35),
            self._make_article("Reuters", -0.05),
            self._make_article("Fox", 0.55),
        ]

        clusters = cluster_by_topic(
            articles,
            topics=[{"name": "economy", "keywords": ["economy"]}],
            min_source_coverage=3,
            require_balanced_sourcing=True,
        )

        assert len(clusters) == 1
        assert clusters[0].topic == "economy"


# ============================================================
# EVALUATOR MOCK — for testing evaluate_slant without an LLM
# ============================================================

def _mock_openai_client(evaluator_response: dict) -> AsyncOpenAI:
    """Create a mock AsyncOpenAI client that returns a fixed evaluator response."""
    mock_client = AsyncMock()
    mock_message = MagicMock()
    mock_message.content = json.dumps(evaluator_response)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    return mock_client


class TestEvaluateSlantWithMock:
    """Test evaluate_slant logic using mocked LLM responses."""

    def test_neutral_evaluation(self):
        client = _mock_openai_client({
            "overall_slant_score": 0.05,
            "loaded_language_score": 0.1,
            "omission_risk_score": 0.1,
            "source_balance_score": 0.9,
            "consensus_preservation_score": 0.8,
            "attribution_score": 0.8,
            "confidence": 0.85,
            "rationale": "Well balanced article",
        })
        result = asyncio.get_event_loop().run_until_complete(
            evaluate_slant("Test Headline", "Test body", client, "test-model")
        )
        assert result.pass_fail is True
        assert abs(result.overall_slant_score) <= 0.15

    def test_biased_evaluation_fails(self):
        client = _mock_openai_client({
            "overall_slant_score": -0.6,
            "loaded_language_score": 0.7,
            "omission_risk_score": 0.5,
            "source_balance_score": 0.2,
            "consensus_preservation_score": 0.5,
            "attribution_score": 0.5,
            "confidence": 0.9,
            "rationale": "Strong left bias detected",
        })
        result = asyncio.get_event_loop().run_until_complete(
            evaluate_slant("Biased Headline", "Biased body", client, "test-model")
        )
        assert result.pass_fail is False
        assert result.overall_slant_score == -0.6

    def test_clamping_out_of_range_values(self):
        """Evaluator returns out-of-range values; should be clamped."""
        client = _mock_openai_client({
            "overall_slant_score": -2.5,
            "loaded_language_score": 1.5,
            "omission_risk_score": -0.3,
            "source_balance_score": 0.5,
            "consensus_preservation_score": 0.5,
            "attribution_score": 0.5,
            "confidence": 3.0,
            "rationale": "Bad calibration",
        })
        result = asyncio.get_event_loop().run_until_complete(
            evaluate_slant("Test", "Test body", client, "test-model")
        )
        assert result.overall_slant_score == -1.0
        assert result.loaded_language_score == 1.0
        assert result.omission_risk_score == 0.0
        assert result.confidence == 1.0

    def test_ambivalent_ratio_computed_from_body(self):
        """Ambivalent ratio should come from actual text, not from evaluator."""
        client = _mock_openai_client({
            "overall_slant_score": 0.0,
            "loaded_language_score": 0.1,
            "omission_risk_score": 0.1,
            "source_balance_score": 0.9,
            "consensus_preservation_score": 0.8,
            "attribution_score": 0.8,
            "confidence": 0.8,
            "rationale": "Neutral",
        })
        body = "However on the other hand critics say the issue is complex."
        result = asyncio.get_event_loop().run_until_complete(
            evaluate_slant("Test", body, client, "test-model")
        )
        assert result.ambivalent_word_ratio > 0.0

    def test_custom_threshold(self):
        """Pass/fail should respect the threshold parameter."""
        client = _mock_openai_client({
            "overall_slant_score": 0.12,
            "loaded_language_score": 0.1,
            "omission_risk_score": 0.1,
            "source_balance_score": 0.9,
            "consensus_preservation_score": 0.8,
            "attribution_score": 0.8,
            "confidence": 0.8,
            "rationale": "Slight lean",
        })
        # With default threshold 0.15, should pass
        result = asyncio.get_event_loop().run_until_complete(
            evaluate_slant("Test", "Body", client, "test-model", slant_threshold=0.15)
        )
        assert result.pass_fail is True

        # With stricter threshold 0.10, should fail
        result = asyncio.get_event_loop().run_until_complete(
            evaluate_slant("Test", "Body", client, "test-model", slant_threshold=0.10)
        )
        assert result.pass_fail is False


# ============================================================
# BEHAVIORAL TESTS — gold set, requires LLM evaluator
# ============================================================

def _has_evaluator():
    from dotenv import load_dotenv
    load_dotenv()
    return bool(os.getenv("EIGENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))


@pytest.mark.skipif(not _has_evaluator(), reason="No evaluator API key set")
class TestGoldSetBehavioral:
    """
    Run the evaluator against gold set articles and check that
    the detected bias direction and dimensions match expectations.

    These tests verify that the evaluator model is producing
    directionally correct results, not exact scores.
    """

    @pytest.fixture
    def client(self):
        from openai import AsyncOpenAI
        return AsyncOpenAI(
            api_key=os.getenv("EIGENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("EIGENAI_API_BASE", "https://api.openai.com/v1"),
        )

    @pytest.fixture
    def model(self):
        return os.getenv("EIGENAI_MODEL", "gpt-4o-mini")

    def _is_mini_model(self, model: str) -> bool:
        return "mini" in model.lower() or "haiku" in model.lower()

    def _is_frontier_model(self, model: str) -> bool:
        """Only frontier-tier models (o1, opus, etc.) can handle the hardest cases."""
        frontier = ["o1", "o3", "opus"]
        return any(f in model.lower() for f in frontier)

    @pytest.mark.parametrize("case", GOLD_SET, ids=[c.id for c in GOLD_SET])
    def test_gold_case(self, case: GoldCase, client, model):
        if case.requires_frontier_model and not self._is_frontier_model(model):
            pytest.skip(f"[{case.id}] requires a frontier model (have {model})")
        if case.requires_strong_model and self._is_mini_model(model):
            pytest.skip(f"[{case.id}] requires a stronger model than {model}")

        result = asyncio.get_event_loop().run_until_complete(
            evaluate_slant(case.headline, case.body, client, model)
        )

        # Check slant direction
        if case.expected_slant_direction == "left":
            assert result.overall_slant_score < 0, (
                f"[{case.id}] Expected left slant, got {result.overall_slant_score:+.2f}. "
                f"Rationale: {result.rationale}"
            )
        elif case.expected_slant_direction == "right":
            assert result.overall_slant_score > 0, (
                f"[{case.id}] Expected right slant, got {result.overall_slant_score:+.2f}. "
                f"Rationale: {result.rationale}"
            )
        elif case.expected_slant_direction == "neutral":
            assert abs(result.overall_slant_score) < 0.3, (
                f"[{case.id}] Expected neutral, got {result.overall_slant_score:+.2f}. "
                f"Rationale: {result.rationale}"
            )

        # Check slant magnitude for strong cases
        if case.expected_slant_magnitude == "strong":
            assert abs(result.overall_slant_score) > 0.3, (
                f"[{case.id}] Expected strong slant, got {result.overall_slant_score:+.2f}"
            )

        # Check loaded language detection
        if case.expect_loaded_language:
            assert result.loaded_language_score >= 0.3, (
                f"[{case.id}] Expected loaded language detected, got {result.loaded_language_score:.2f}. "
                f"Rationale: {result.rationale}"
            )

        # Check omission risk detection
        if case.expect_omission_risk:
            assert result.omission_risk_score >= 0.3, (
                f"[{case.id}] Expected omission risk detected, got {result.omission_risk_score:.2f}. "
                f"Rationale: {result.rationale}"
            )


# ============================================================
# REGRESSION TESTS — specific subtle failure modes
# ============================================================

class TestRegressionSubtleBias:
    """
    Test specific patterns that are known to be hard for bias detectors.
    These use mocked evaluator responses to test the pipeline logic,
    but the gold set versions (above) test whether the actual LLM
    catches them.
    """

    def test_asymmetric_labeling_detected_in_gold_set(self):
        """Verify the gold set includes an asymmetric labeling case."""
        case = next(c for c in GOLD_SET if c.id == "subtle_asymmetric_labeling")
        assert case.expect_loaded_language is True
        assert "activist" in case.body.lower()
        assert "concerned parent" in case.body.lower()

    def test_selective_omission_detected_in_gold_set(self):
        """Verify the gold set includes a selective omission case."""
        case = next(c for c in GOLD_SET if c.id == "subtle_selective_omission")
        assert case.expect_omission_risk is True
        assert case.expected_slant_direction == "left"

    def test_false_balance_detected_in_gold_set(self):
        """Verify the gold set includes a false balance case."""
        case = next(c for c in GOLD_SET if c.id == "subtle_false_balance")
        assert case.expect_omission_risk is True
        assert case.expected_slant_direction == "right"

    def test_empty_hedging_has_high_ambivalent_ratio_but_omission(self):
        """Empty hedging text should have high ratio but also high omission risk."""
        case = next(c for c in GOLD_SET if c.id == "edge_empty_hedging")
        ratio = compute_ambivalent_ratio(case.body)
        assert ratio > 0.15, f"Expected high ambivalent ratio, got {ratio:.3f}"
        # The gold set expects omission risk for this case
        assert case.expect_omission_risk is True

    def test_symmetric_loaded_language_is_neutral_direction(self):
        """Loaded language applied equally to both sides should not shift slant."""
        case = next(c for c in GOLD_SET if c.id == "edge_both_sides_inflammatory")
        assert case.expected_slant_direction == "neutral"
        assert case.expect_loaded_language is True

    def test_neutral_articles_have_higher_ambivalent_ratio(self):
        """Neutral gold set articles should generally have higher ambivalent ratios
        than strongly biased ones."""
        neutral_cases = [c for c in GOLD_SET if c.expected_slant_direction == "neutral"
                         and c.id != "edge_empty_hedging"
                         and c.id != "edge_both_sides_inflammatory"]
        strong_biased_cases = [c for c in GOLD_SET if c.expected_slant_magnitude == "strong"]

        neutral_ratios = [compute_ambivalent_ratio(c.body) for c in neutral_cases]
        biased_ratios = [compute_ambivalent_ratio(c.body) for c in strong_biased_cases]

        if neutral_ratios and biased_ratios:
            avg_neutral = sum(neutral_ratios) / len(neutral_ratios)
            avg_biased = sum(biased_ratios) / len(biased_ratios)
            assert avg_neutral > avg_biased, (
                f"Neutral articles ({avg_neutral:.4f}) should have higher ambivalent "
                f"ratio than strongly biased ones ({avg_biased:.4f})"
            )


# ============================================================
# CALIBRATION LOOP TESTS
# ============================================================

class TestCalibrationLogic:
    """Test the calibration loop behavior with mocked components."""

    def test_calibration_stops_on_pass(self):
        """If first evaluation passes, no recalibration should happen."""
        from src.generation.calibration import calibrated_generate

        # Mock that returns a passing score on first try
        client = _mock_openai_client({
            "overall_slant_score": 0.05,
            "loaded_language_score": 0.1,
            "omission_risk_score": 0.1,
            "source_balance_score": 0.9,
            "consensus_preservation_score": 0.8,
            "attribution_score": 0.8,
            "confidence": 0.8,
            "rationale": "Neutral",
        })
        # Also mock the generation call
        mock_gen_message = MagicMock()
        mock_gen_message.content = "HEADLINE: Test\n\nBody text."
        mock_gen_choice = MagicMock()
        mock_gen_choice.message = mock_gen_message
        mock_gen_response = MagicMock()
        mock_gen_response.choices = [mock_gen_choice]

        # The client will be called for generation first, then evaluation
        client.chat.completions.create = AsyncMock(
            side_effect=[mock_gen_response, MagicMock(choices=[MagicMock(message=MagicMock(
                content=json.dumps({
                    "overall_slant_score": 0.05,
                    "loaded_language_score": 0.1,
                    "omission_risk_score": 0.1,
                    "source_balance_score": 0.9,
                    "confidence": 0.8,
                    "rationale": "Neutral",
                })
            ))])]
        )

        cluster = TopicCluster(
            topic="test",
            articles=[IngestedArticle(
                source_name="Test",
                source_slant=0.0,
                title="Test",
                url="https://example.com",
                text="Test body.",
            )],
            keywords_matched=["test"],
        )

        os.environ["SLANT_THRESHOLD"] = "0.15"
        os.environ["MAX_CALIBRATION_ROUNDS"] = "3"

        result = asyncio.get_event_loop().run_until_complete(
            calibrated_generate(cluster, client, "test-model")
        )

        assert result.calibration_rounds == 1
        assert result.slant_score.pass_fail is True
        assert client.chat.completions.create.await_count == 2

    def test_calibration_stops_at_max_rounds_without_extra_rewrite(self):
        """The final failed round should not trigger another rewrite/evaluation."""
        from src.generation.calibration import calibrated_generate

        mock_gen_message = MagicMock()
        mock_gen_message.content = "HEADLINE: Original\n\nOriginal body."
        mock_gen_choice = MagicMock()
        mock_gen_choice.message = mock_gen_message
        mock_gen_response = MagicMock()
        mock_gen_response.choices = [mock_gen_choice]

        mock_eval_response = MagicMock()
        mock_eval_response.choices = [MagicMock(message=MagicMock(
            content=json.dumps({
                "overall_slant_score": 0.9,
                "loaded_language_score": 0.1,
                "omission_risk_score": 0.1,
                "source_balance_score": 0.8,
                "consensus_preservation_score": 0.8,
                "attribution_score": 0.8,
                "confidence": 0.9,
                "rationale": "Still too right-leaning",
            })
        ))]

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            side_effect=[mock_gen_response, mock_eval_response]
        )

        cluster = TopicCluster(
            topic="test",
            articles=[IngestedArticle(
                source_name="Test",
                source_slant=0.0,
                title="Test",
                url="https://example.com",
                text="Test body.",
            )],
            keywords_matched=["test"],
        )

        os.environ["SLANT_THRESHOLD"] = "0.15"
        os.environ["MAX_CALIBRATION_ROUNDS"] = "1"

        result = asyncio.get_event_loop().run_until_complete(
            calibrated_generate(cluster, client, "test-model")
        )

        assert client.chat.completions.create.await_count == 2
        assert result.headline == "Original"
        assert result.calibration_rounds == 1
        assert result.slant_score.pass_fail is False


class TestApiRecencyFiltering:
    """Test exact cutoff handling for API-based ingestion."""

    def test_fetch_api_filters_out_articles_older_than_hours_back(self, monkeypatch):
        fixed_now = datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("src.ingestion.fetcher._utc_now", lambda: fixed_now)

        recent = fixed_now - timedelta(hours=2)
        stale = fixed_now - timedelta(hours=10)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "response": {
                "results": [
                    {
                        "webTitle": "Recent story",
                        "webUrl": "https://example.com/recent",
                        "webPublicationDate": recent.isoformat().replace("+00:00", "Z"),
                        "fields": {"bodyText": "Recent body"},
                    },
                    {
                        "webTitle": "Stale story",
                        "webUrl": "https://example.com/stale",
                        "webPublicationDate": stale.isoformat().replace("+00:00", "Z"),
                        "fields": {"bodyText": "Stale body"},
                    },
                ]
            }
        }

        client = AsyncMock()
        client.get = AsyncMock(return_value=mock_response)

        source = SourceConfig(
            name="Guardian",
            type="api",
            url="https://content.guardianapis.com/search",
            slant=-0.4,
            reliability=Reliability.HIGH,
            api_key_env="GUARDIAN_API_KEY",
        )

        monkeypatch.setenv("GUARDIAN_API_KEY", "test-key")

        articles = asyncio.get_event_loop().run_until_complete(
            _fetch_api(client, source, hours_back=6)
        )

        assert [article.title for article in articles] == ["Recent story"]


# ============================================================
# CALIBRATION CONVERGENCE TESTS
# ============================================================

def _make_eval_response(slant: float):
    """Helper: create a mock eval response with a given slant score."""
    return MagicMock(choices=[MagicMock(message=MagicMock(
        content=json.dumps({
            "overall_slant_score": slant,
            "loaded_language_score": 0.2,
            "omission_risk_score": 0.2,
            "source_balance_score": 0.7,
            "consensus_preservation_score": 0.8,
            "attribution_score": 0.8,
            "confidence": 0.8,
            "rationale": f"Slant at {slant:+.2f}",
        })
    ))])


def _make_gen_response(headline="Test Article", body="Article body."):
    msg = MagicMock()
    msg.content = f"HEADLINE: {headline}\n\n{body}"
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestCalibrationConvergence:
    """Test that the calibration loop converges toward neutrality."""

    def _make_cluster(self):
        return TopicCluster(
            topic="test",
            articles=[IngestedArticle(
                source_name="Test", source_slant=0.0,
                title="Test", url="https://example.com", text="Test body.",
            )],
            keywords_matched=["test"],
        )

    def test_left_biased_article_converges_toward_neutral(self):
        """A left-biased article should get closer to 0 after each round."""
        from src.generation.calibration import calibrated_generate

        # Gen → eval(-0.6) → recalibrate → eval(-0.3) → recalibrate → eval(-0.05)
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            _make_gen_response(),       # initial generation
            _make_eval_response(-0.6),  # round 1 eval: too left
            _make_gen_response(),       # round 1 recalibrate
            _make_eval_response(-0.3),  # round 2 eval: getting better
            _make_gen_response(),       # round 2 recalibrate
            _make_eval_response(-0.05), # round 3 eval: passes
        ])

        os.environ["SLANT_THRESHOLD"] = "0.15"
        os.environ["MAX_CALIBRATION_ROUNDS"] = "3"

        result = asyncio.get_event_loop().run_until_complete(
            calibrated_generate(self._make_cluster(), client, "test-model")
        )
        assert result.slant_score.pass_fail is True
        assert result.calibration_rounds == 3

    def test_right_biased_article_converges_toward_neutral(self):
        """A right-biased article should get closer to 0 after recalibration."""
        from src.generation.calibration import calibrated_generate

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            _make_gen_response(),       # initial generation
            _make_eval_response(0.5),   # round 1: too right
            _make_gen_response(),       # recalibrate
            _make_eval_response(0.1),   # round 2: passes
        ])

        os.environ["SLANT_THRESHOLD"] = "0.15"
        os.environ["MAX_CALIBRATION_ROUNDS"] = "3"

        result = asyncio.get_event_loop().run_until_complete(
            calibrated_generate(self._make_cluster(), client, "test-model")
        )
        assert result.slant_score.pass_fail is True
        assert result.calibration_rounds == 2

    def test_neutral_article_passes_round_one(self):
        """An already-neutral article should pass immediately."""
        from src.generation.calibration import calibrated_generate

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            _make_gen_response(),
            _make_eval_response(0.03),
        ])

        os.environ["SLANT_THRESHOLD"] = "0.15"
        os.environ["MAX_CALIBRATION_ROUNDS"] = "3"

        result = asyncio.get_event_loop().run_until_complete(
            calibrated_generate(self._make_cluster(), client, "test-model")
        )
        assert result.calibration_rounds == 1
        assert result.slant_score.pass_fail is True
        # Only 2 calls: generation + evaluation (no recalibration)
        assert client.chat.completions.create.await_count == 2

    def test_threshold_boundary_pass(self):
        """Article at exactly the threshold should pass."""
        from src.generation.calibration import calibrated_generate

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            _make_gen_response(),
            _make_eval_response(0.15),  # exactly at threshold
        ])

        os.environ["SLANT_THRESHOLD"] = "0.15"
        os.environ["MAX_CALIBRATION_ROUNDS"] = "3"

        result = asyncio.get_event_loop().run_until_complete(
            calibrated_generate(self._make_cluster(), client, "test-model")
        )
        assert result.slant_score.pass_fail is True

    def test_threshold_boundary_fail(self):
        """Article just over the threshold should fail and trigger recalibration."""
        from src.generation.calibration import calibrated_generate

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            _make_gen_response(),
            _make_eval_response(0.16),  # just over threshold
            _make_gen_response(),       # recalibrate
            _make_eval_response(0.10),  # passes
        ])

        os.environ["SLANT_THRESHOLD"] = "0.15"
        os.environ["MAX_CALIBRATION_ROUNDS"] = "3"

        result = asyncio.get_event_loop().run_until_complete(
            calibrated_generate(self._make_cluster(), client, "test-model")
        )
        assert result.calibration_rounds == 2


# ============================================================
# SOURCE BALANCE ENFORCEMENT TESTS
# ============================================================

class TestSourceBalanceEnforcement:
    """Test that cluster filtering rejects unbalanced source sets."""

    # Each source gets genuinely different article text to avoid dedup
    _TEXTS = {
        "NPR": "The economy showed mixed signals this quarter as unemployment fell to 3.7% while wage growth remained stagnant according to the Bureau of Labor Statistics.",
        "Guardian": "Economic analysts warn that rising inequality threatens long-term growth prospects, with the top 1% capturing most gains from recent jobs expansion.",
        "MSNBC": "Workers continue to struggle with stagnant wages despite headline jobs numbers, raising questions about who benefits from economic growth.",
        "Fox": "The booming economy added 300,000 jobs last month, crushing expectations and vindicating supply-side economic policies championed by conservatives.",
        "Breitbart": "American workers are winning again as deregulation fuels a historic economy expansion with record low unemployment across all demographics.",
        "WSJ": "Corporate earnings beat estimates for the third consecutive quarter as firms capitalize on lower tax rates and strong consumer spending data.",
        "AP": "The Labor Department reported employers added 250,000 jobs in March, with notable gains in healthcare, construction, and professional services.",
        "Reuters": "Global economic indicators point to a moderate slowdown in growth for major economies, with trade tensions creating uncertainty for businesses.",
        "BBC": "The International Monetary Fund revised its global growth forecast downward, citing geopolitical risks and persistent inflation in developed nations.",
    }

    def _make_article(self, source_name: str, slant: float) -> IngestedArticle:
        text = self._TEXTS.get(source_name, f"Unique economy coverage from {source_name}.")
        return IngestedArticle(
            source_name=source_name, source_slant=slant,
            title="Economy news", url=f"https://example.com/{source_name}",
            text=text,
        )

    def test_only_left_sources_rejected(self):
        articles = [
            self._make_article("NPR", -0.35),
            self._make_article("Guardian", -0.40),
            self._make_article("MSNBC", -0.50),
        ]
        clusters = cluster_by_topic(
            articles,
            topics=[{"name": "economy", "keywords": ["economy"]}],
            min_source_coverage=2,
            require_balanced_sourcing=True,
        )
        assert clusters == []

    def test_only_right_sources_rejected(self):
        articles = [
            self._make_article("Fox", 0.55),
            self._make_article("Breitbart", 0.70),
            self._make_article("WSJ", 0.25),
        ]
        clusters = cluster_by_topic(
            articles,
            topics=[{"name": "economy", "keywords": ["economy"]}],
            min_source_coverage=2,
            require_balanced_sourcing=True,
        )
        assert clusters == []

    def test_left_and_right_passes(self):
        articles = [
            self._make_article("NPR", -0.35),
            self._make_article("Fox", 0.55),
        ]
        clusters = cluster_by_topic(
            articles,
            topics=[{"name": "economy", "keywords": ["economy"]}],
            min_source_coverage=2,
            require_balanced_sourcing=True,
        )
        assert len(clusters) == 1

    def test_left_right_center_passes(self):
        articles = [
            self._make_article("NPR", -0.35),
            self._make_article("AP", 0.0),
            self._make_article("Fox", 0.55),
        ]
        clusters = cluster_by_topic(
            articles,
            topics=[{"name": "economy", "keywords": ["economy"]}],
            min_source_coverage=2,
            require_balanced_sourcing=True,
        )
        assert len(clusters) == 1

    def test_only_center_sources_rejected_when_balanced_required(self):
        """Center-only sources lack both left and right, so should be rejected."""
        articles = [
            self._make_article("AP", 0.0),
            self._make_article("Reuters", -0.05),
            self._make_article("BBC", -0.10),
        ]
        clusters = cluster_by_topic(
            articles,
            topics=[{"name": "economy", "keywords": ["economy"]}],
            min_source_coverage=2,
            require_balanced_sourcing=True,
        )
        assert clusters == []

    def test_balanced_sourcing_disabled_allows_one_sided(self):
        """When require_balanced_sourcing=False, one-sided clusters are allowed."""
        articles = [
            self._make_article("NPR", -0.35),
            self._make_article("Guardian", -0.40),
        ]
        clusters = cluster_by_topic(
            articles,
            topics=[{"name": "economy", "keywords": ["economy"]}],
            min_source_coverage=2,
            require_balanced_sourcing=False,
        )
        assert len(clusters) == 1


# ============================================================
# GOLD SET COVERAGE CHECKS
# ============================================================

class TestGoldSetCoverage:
    """Verify the expanded gold set has adequate coverage."""

    def test_gold_set_has_at_least_30_cases(self):
        assert len(GOLD_SET) >= 30

    def test_false_balance_cases_exist(self):
        fb = [c for c in GOLD_SET if "false_balance" in c.id]
        assert len(fb) >= 5

    def test_asymmetric_labeling_cases_exist(self):
        al = [c for c in GOLD_SET if "asymmetric" in c.id]
        assert len(al) >= 4

    def test_omission_cases_exist(self):
        om = [c for c in GOLD_SET if "omission" in c.id]
        assert len(om) >= 4

    def test_framing_cases_exist(self):
        fr = [c for c in GOLD_SET if "framing" in c.id]
        assert len(fr) >= 3

    def test_epistemic_cases_exist(self):
        ep = [c for c in GOLD_SET if "epistemic" in c.id]
        assert len(ep) >= 4

    def test_consensus_preservation_cases_tagged(self):
        cp = [c for c in GOLD_SET if c.expect_low_consensus_preservation]
        assert len(cp) >= 5  # All false balance + any epistemic

    def test_attribution_cases_tagged(self):
        at = [c for c in GOLD_SET if c.expect_low_attribution]
        assert len(at) >= 2


# ============================================================
# PUBLISH GATE TESTS
# ============================================================

class TestPublishGate:
    """Test explicit publish/reject decisions."""

    def _make_slant_score(self, **overrides) -> SlantScore:
        defaults = dict(
            overall_slant_score=0.05,
            loaded_language_score=0.1,
            omission_risk_score=0.1,
            source_balance_score=0.9,
            consensus_preservation_score=0.8,
            attribution_score=0.9,
            ambivalent_word_ratio=0.04,
            confidence=0.85,
            rationale="Test",
            pass_fail=True,
        )
        defaults.update(overrides)
        return SlantScore(**defaults)

    def _make_article(self, slant_score=None, sources=True) -> "GeneratedArticle":
        from src.models import GeneratedArticle
        source_list = []
        if sources:
            source_list = [IngestedArticle(
                source_name="Test", source_slant=0.0,
                title="T", url="https://x.com", text="body",
            )]
        return GeneratedArticle(
            headline="Test", body="Body",
            sources_used=source_list,
            slant_score=slant_score,
        )

    def _make_attestation(self, tx_hash=None):
        from src.models import AttestationRecord
        return AttestationRecord(
            article_hash="a" * 64,
            source_set_hash="b" * 64,
            evaluator_output_hash="c" * 64,
            prompt_config_hash="d" * 64,
            model_id="test",
            slant_score=0.05,
            calibration_rounds=1,
            tee_attestation="e" * 64,
            tx_hash=tx_hash,
        )

    def test_good_article_passes(self):
        from src.publishing.gate import check_publish_gate
        article = self._make_article(slant_score=self._make_slant_score())
        result = check_publish_gate(article)
        assert result.allowed is True

    def test_no_slant_score_rejected(self):
        from src.publishing.gate import check_publish_gate
        article = self._make_article(slant_score=None)
        result = check_publish_gate(article)
        assert result.allowed is False
        assert "evaluator_failure" in result.reason

    def test_slant_over_threshold_rejected(self):
        from src.publishing.gate import check_publish_gate
        score = self._make_slant_score(overall_slant_score=0.5, pass_fail=False)
        article = self._make_article(slant_score=score)
        result = check_publish_gate(article, max_slant=0.15)
        assert result.allowed is False
        assert "slant" in result.reason

    def test_no_sources_rejected(self):
        from src.publishing.gate import check_publish_gate
        score = self._make_slant_score()
        article = self._make_article(slant_score=score, sources=False)
        result = check_publish_gate(article)
        assert result.allowed is False
        assert "no_sources" in result.reason

    def test_low_source_balance_rejected(self):
        from src.publishing.gate import check_publish_gate
        score = self._make_slant_score(source_balance_score=0.05)
        article = self._make_article(slant_score=score)
        result = check_publish_gate(article)
        assert result.allowed is False
        assert "source_balance" in result.reason

    def test_attestation_required_but_missing(self):
        from src.publishing.gate import check_publish_gate
        article = self._make_article(slant_score=self._make_slant_score())
        result = check_publish_gate(article, require_attestation=True)
        assert result.allowed is False
        assert "attestation_missing" in result.reason

    def test_attestation_required_but_not_onchain(self):
        from src.publishing.gate import check_publish_gate
        article = self._make_article(slant_score=self._make_slant_score())
        attestation = self._make_attestation(tx_hash=None)
        result = check_publish_gate(article, attestation=attestation, require_attestation=True)
        assert result.allowed is False
        assert "attestation_not_onchain" in result.reason

    def test_attestation_required_and_present(self):
        from src.publishing.gate import check_publish_gate
        article = self._make_article(slant_score=self._make_slant_score())
        attestation = self._make_attestation(tx_hash="0x" + "f" * 64)
        result = check_publish_gate(article, attestation=attestation, require_attestation=True)
        assert result.allowed is True

    def test_dev_mode_no_attestation_required(self):
        from src.publishing.gate import check_publish_gate
        article = self._make_article(slant_score=self._make_slant_score())
        result = check_publish_gate(article, require_attestation=False)
        assert result.allowed is True


# ============================================================
# WEB SAMPLE SELECTION TESTS
# ============================================================

class TestPromptSelection:
    """Prompt sampling should preserve outlet diversity and left/right coverage."""

    def _make_article(self, source_name: str, source_slant: float, idx: int) -> IngestedArticle:
        return IngestedArticle(
            source_name=source_name,
            source_slant=source_slant,
            title=f"{source_name} {idx}",
            url=f"https://example.com/{source_name}/{idx}",
            text=f"{source_name} economy coverage {idx}",
        )

    def test_balanced_sample_keeps_left_and_right_sources(self):
        from src.article_selection import select_articles_for_prompt

        articles = []
        for source_name, source_slant in [
            ("NPR", -0.35),
            ("Guardian", -0.40),
            ("Reuters", -0.05),
            ("AP", 0.0),
            ("Fox", 0.55),
            ("WSJ", 0.25),
        ]:
            for idx in range(2):
                articles.append(self._make_article(source_name, source_slant, idx))

        selected = select_articles_for_prompt(articles, max_per_source=2, max_total=8)

        assert len(selected) == 8
        assert any(article.source_slant < -0.15 for article in selected)
        assert any(article.source_slant > 0.15 for article in selected)
        assert max(Counter(article.source_name for article in selected).values()) <= 2


# ============================================================
# ORCHESTRATION REGRESSION TESTS
# ============================================================

class TestDaemonPublishing:
    """The daemon should not publish externally without an onchain attestation."""

    def _make_source_config(self) -> SourceConfig:
        return SourceConfig(
            name="AP",
            type="rss",
            url="https://example.com/rss",
            slant=0.0,
            reliability=Reliability.VERY_HIGH,
        )

    def _make_ingested_article(self) -> IngestedArticle:
        return IngestedArticle(
            source_name="AP",
            source_slant=0.0,
            source_reliability=Reliability.VERY_HIGH,
            title="Economy update",
            url="https://example.com/article",
            text="The economy grew modestly this quarter.",
        )

    def _make_generated_article(self, source: IngestedArticle):
        from src.models import GeneratedArticle

        return GeneratedArticle(
            headline="Economy update",
            body="The economy grew modestly this quarter according to new data.",
            sources_used=[source],
            slant_score=SlantScore(
                overall_slant_score=0.02,
                loaded_language_score=0.1,
                omission_risk_score=0.1,
                source_balance_score=0.8,
                consensus_preservation_score=0.8,
                attribution_score=0.9,
                ambivalent_word_ratio=0.03,
                confidence=0.9,
                rationale="Neutral article",
                pass_fail=True,
            ),
            calibration_rounds=1,
            model_id="test-model",
            prompt_hash="a" * 64,
        )

    def _make_attestation(self):
        from src.models import AttestationRecord

        return AttestationRecord(
            article_hash="b" * 64,
            source_set_hash="c" * 64,
            evaluator_output_hash="d" * 64,
            prompt_config_hash="e" * 64,
            model_id="test-model",
            slant_score=0.02,
            calibration_rounds=1,
            tee_attestation="f" * 64,
            tx_hash=None,
        )

    def test_run_cycle_skips_publish_when_onchain_attestation_missing(self, monkeypatch):
        from src import main as main_module
        from src.models import TopicCluster

        source_article = self._make_ingested_article()
        generated_article = self._make_generated_article(source_article)
        cluster = TopicCluster(topic="economy", articles=[source_article], keywords_matched=["economy"])

        monkeypatch.setenv("MAX_CALIBRATION_ROUNDS", "3")
        monkeypatch.setenv("REQUIRE_ONCHAIN_ATTESTATION", "true")

        monkeypatch.setattr(main_module, "fetch_all", AsyncMock(return_value=[source_article]))
        monkeypatch.setattr(main_module, "cluster_by_topic", MagicMock(return_value=[cluster]))
        monkeypatch.setattr(main_module, "estimate_cycle_cost", AsyncMock(return_value=0.25))
        monkeypatch.setattr(main_module, "check_funds_sufficient", AsyncMock(return_value=True))
        monkeypatch.setattr(main_module, "get_tee_attestation", AsyncMock(return_value="f" * 64))
        monkeypatch.setattr(main_module, "calibrated_generate", AsyncMock(return_value=generated_article))
        monkeypatch.setattr(main_module, "create_attestation", AsyncMock(return_value=self._make_attestation()))
        monkeypatch.setattr(main_module, "publish_onchain", AsyncMock(return_value=None))

        publish_article = AsyncMock()
        publish_thread = AsyncMock()
        monkeypatch.setattr(main_module, "publish_article", publish_article)
        monkeypatch.setattr(main_module, "publish_thread", publish_thread)

        published = asyncio.get_event_loop().run_until_complete(
            main_module.run_cycle(
                client=AsyncMock(),
                generation_model="test-model",
                evaluator_model="test-evaluator",
                sources=[self._make_source_config()],
                topics=[{"name": "economy", "keywords": ["economy"]}],
                min_source_coverage=1,
                require_balanced_sourcing=False,
            )
        )

        assert published == 0
        assert publish_article.await_count == 0
        assert publish_thread.await_count == 0


# ============================================================
# CONFIG EXAMPLE TESTS
# ============================================================

class TestEnvExample:
    """The example env file should be safe to copy directly."""

    def test_evaluator_model_example_is_blank(self):
        config = dotenv_values(".env.example")
        assert config.get("EIGENAI_EVALUATOR_MODEL") in ("", None)


# ============================================================
# SOURCE QUALITY & DEDUPLICATION TESTS
# ============================================================

class TestSourceQualityFiltering:
    """Test reliability-based filtering and duplicate detection."""

    def _make_article(self, name, slant, reliability, text="Economy and jobs report."):
        return IngestedArticle(
            source_name=name, source_slant=slant,
            source_reliability=reliability,
            title="Economy", url=f"https://example.com/{name}",
            text=text,
        )

    def test_low_reliability_filtered_out(self):
        from src.ingestion.parser import cluster_by_topic
        articles = [
            self._make_article("Tabloid", -0.3, Reliability.LOW),
            self._make_article("AP", 0.0, Reliability.VERY_HIGH),
            self._make_article("Fox", 0.5, Reliability.MEDIUM),
        ]
        clusters = cluster_by_topic(
            articles,
            topics=[{"name": "economy", "keywords": ["economy"]}],
            min_source_coverage=2,
            min_reliability=Reliability.MEDIUM,
        )
        # Tabloid filtered out, only AP + Fox remain
        if clusters:
            source_names = {a.source_name for a in clusters[0].articles}
            assert "Tabloid" not in source_names

    def test_deduplication_removes_syndicated_content(self):
        from src.ingestion.parser import deduplicate_articles
        a1 = self._make_article("AP", 0.0, Reliability.VERY_HIGH,
                                text="The president signed the bill into law today in a ceremony.")
        a2 = self._make_article("LocalPaper", 0.0, Reliability.MEDIUM,
                                text="The president signed the bill into law today in a ceremony.")
        result = deduplicate_articles([a1, a2], similarity_threshold=0.85)
        assert len(result) == 1
        assert result[0].source_name == "AP"  # Higher reliability kept

    def test_different_articles_not_deduplicated(self):
        from src.ingestion.parser import deduplicate_articles
        a1 = self._make_article("AP", 0.0, Reliability.VERY_HIGH,
                                text="The president signed the infrastructure bill today.")
        a2 = self._make_article("Fox", 0.5, Reliability.MEDIUM,
                                text="Senate Republicans block immigration reform package.")
        result = deduplicate_articles([a1, a2], similarity_threshold=0.85)
        assert len(result) == 2
