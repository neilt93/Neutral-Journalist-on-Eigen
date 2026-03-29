"""
Publish gate — explicit go/no-go decisions before publishing an article.

Checks ALL bias and epistemic dimensions, not just slant. A false-balance
article with 0.00 slant but terrible consensus preservation gets blocked.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import structlog

from src.models import GeneratedArticle, AttestationRecord

log = structlog.get_logger()


@dataclass
class GateResult:
    allowed: bool
    reason: str
    failures: list[str] = field(default_factory=list)


def check_publish_gate(
    article: GeneratedArticle,
    attestation: AttestationRecord | None = None,
    max_slant: float | None = None,
    max_loaded_language: float | None = None,
    max_omission_risk: float | None = None,
    min_source_balance: float | None = None,
    min_consensus_preservation: float | None = None,
    min_attribution: float | None = None,
    require_attestation: bool = False,
) -> GateResult:
    """
    Check whether an article is safe to publish.
    Rejects if ANY dimension exceeds its threshold.
    """
    # Load thresholds from env with sensible defaults
    slant_t = max_slant if max_slant is not None else float(os.getenv("SLANT_THRESHOLD", "0.15"))
    loaded_t = max_loaded_language if max_loaded_language is not None else float(os.getenv("LOADED_LANGUAGE_THRESHOLD", "0.7"))
    omission_t = max_omission_risk if max_omission_risk is not None else float(os.getenv("OMISSION_RISK_THRESHOLD", "0.7"))
    balance_t = min_source_balance if min_source_balance is not None else float(os.getenv("SOURCE_BALANCE_THRESHOLD", "0.4"))
    consensus_t = min_consensus_preservation if min_consensus_preservation is not None else float(os.getenv("CONSENSUS_PRESERVATION_THRESHOLD", "0.4"))
    attribution_t = min_attribution if min_attribution is not None else float(os.getenv("ATTRIBUTION_THRESHOLD", "0.4"))

    # 1. Slant score must exist
    if article.slant_score is None:
        return GateResult(
            allowed=False,
            reason="evaluator_failure: no slant score",
            failures=["evaluator_failure"],
        )

    # 2. Must have sources
    if not article.sources_used:
        return GateResult(
            allowed=False,
            reason="no_sources: article has no source articles",
            failures=["no_sources"],
        )

    # 3. Check all dimensions
    s = article.slant_score
    failures = []

    if abs(s.overall_slant_score) > slant_t:
        failures.append(f"slant: {s.overall_slant_score:+.2f} exceeds ±{slant_t}")

    if s.loaded_language_score > loaded_t:
        failures.append(f"loaded_language: {s.loaded_language_score:.2f} exceeds {loaded_t}")

    if s.omission_risk_score > omission_t:
        failures.append(f"omission_risk: {s.omission_risk_score:.2f} exceeds {omission_t}")

    if s.source_balance_score < balance_t:
        failures.append(f"source_balance: {s.source_balance_score:.2f} below {balance_t}")

    if s.consensus_preservation_score < consensus_t:
        failures.append(f"consensus_preservation: {s.consensus_preservation_score:.2f} below {consensus_t}")

    if s.attribution_score < attribution_t:
        failures.append(f"attribution: {s.attribution_score:.2f} below {attribution_t}")

    if failures:
        reason = "gate_failed: " + "; ".join(failures)
        log.warning("publish_gate_failed", failures=failures, headline=article.headline[:60])
        return GateResult(allowed=False, reason=reason, failures=failures)

    # 4. Attestation required in prod mode
    if require_attestation:
        if attestation is None:
            return GateResult(allowed=False, reason="attestation_missing", failures=["attestation_missing"])
        if attestation.tx_hash is None:
            return GateResult(allowed=False, reason="attestation_not_onchain", failures=["attestation_not_onchain"])

    return GateResult(allowed=True, reason="all_checks_passed")
