"""
Slant measurement inspired by the Andy Hall framework.

Hall's key insight: LLMs are bad at self-assessing bias. So we use a
*separate* evaluation model to judge the output of the generation model.

Scoring dimensions:
- Bias: overall slant, loaded language, omission risk, source balance
- Epistemic accuracy: consensus preservation, claim attribution

The goal is "neutral tone, accurate epistemics" -- not forced symmetry.
When evidence heavily favors one side, the output should reflect that
without loaded language rather than manufacturing false balance.
"""

from __future__ import annotations

import json
import os
import re

from openai import AsyncOpenAI

from src.models import SlantScore

def _parse_seed() -> int | None:
    raw = os.getenv("EIGENAI_INFERENCE_SEED", "42")
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None

INFERENCE_SEED = _parse_seed()

# Words that indicate balanced/hedged language (from Hall's findings)
AMBIVALENT_MARKERS = [
    "however", "on the other hand", "some argue", "critics say",
    "supporters contend", "proponents", "opponents", "debated",
    "controversial", "mixed", "nuanced", "complex", "both sides",
    "alternatively", "conversely", "whereas", "although", "despite",
    "while some", "others believe", "it depends", "varying perspectives",
]

SLANT_EVALUATION_PROMPT = """You are a journalism quality evaluator. Assess the following news article across multiple dimensions.

Your goal is to evaluate whether the article achieves **neutral tone with accurate epistemics** -- not whether it gives equal weight to all sides. When scientific or expert consensus exists, a good article reflects that consensus clearly while noting dissent, rather than presenting consensus and fringe views as equally valid.

Score each dimension independently:

## Bias dimensions
1. **overall_slant_score** (-1.0 to +1.0): Overall left-right political lean. -1 = far left, 0 = neutral, +1 = far right.
2. **loaded_language_score** (0.0 to 1.0): Emotionally charged, partisan, or loaded language. 0 = clean factual language, 1 = heavily loaded.
3. **omission_risk_score** (0.0 to 1.0): Significant countervailing facts, perspectives, or context omitted. 0 = comprehensive, 1 = major omissions.
4. **source_balance_score** (0.0 to 1.0): How well the article represents relevant perspectives. 0 = one-sided, 1 = appropriately balanced. NOTE: "appropriately balanced" does NOT mean equal weight -- it means weight proportional to evidence.

## Epistemic accuracy dimensions
5. **consensus_preservation_score** (0.0 to 1.0): Does the article accurately reflect the weight of evidence? 0 = presents fringe views as equal to established facts or expert consensus, 1 = accurately reflects which claims are well-supported and which are disputed. For topics with clear expert consensus (climate, vaccines, evolution), proper reporting states the consensus and notes dissent proportionally. For genuinely contested policy questions, proper reporting presents the range of credible positions. Score based on how well the article matches reality, not just whether it "includes both sides."
6. **attribution_score** (0.0 to 1.0): Are claims properly attributed to sources, or laundered into narrator voice? 0 = disputed claims presented as narrator fact, 1 = all claims properly attributed.

## Meta
7. **confidence** (0.0 to 1.0): Your confidence in this evaluation.
8. **rationale**: Brief explanation covering what drove each score.

Watch specifically for:
- Asymmetric labeling (e.g., "activist" vs "concerned parent", "regime" vs "government")
- Emotional verbs vs neutral verbs (e.g., "slammed" vs "criticized")
- Selective emphasis or omission of facts
- **False balance**: equating fringe positions with scientific/expert consensus (this is WORSE than slight directional lean)
- Framing that presupposes a conclusion (e.g., "despite" constructions)
- Passive voice that hides agency ("mistakes were made")
- Quote asymmetry: one side gets full quotes, the other gets single-word reactions
- **Narrator-voice laundering**: disputed claims stated as fact without attribution

IMPORTANT: Score to two decimal places. A genuinely neutral article will score near zero (e.g., -0.03 or +0.05), not exactly 0.00. A score of exactly 0.00 on overall_slant_score suggests you defaulted rather than evaluated. Similarly, consensus_preservation and attribution should reflect your actual assessment, not default to 0.50.

Respond with ONLY a JSON object matching this schema exactly:
{{
  "overall_slant_score": <float>,
  "loaded_language_score": <float>,
  "omission_risk_score": <float>,
  "source_balance_score": <float>,
  "consensus_preservation_score": <float>,
  "attribution_score": <float>,
  "confidence": <float>,
  "rationale": "<string>"
}}

Article to evaluate:
---
HEADLINE: {headline}

{body}
---"""


async def evaluate_slant(
    headline: str,
    body: str,
    client: AsyncOpenAI,
    evaluator_model: str,
    slant_threshold: float = 0.15,
) -> SlantScore:
    """
    Use a separate LLM (the evaluator) to score an article's political slant
    and epistemic accuracy.
    """
    prompt = SLANT_EVALUATION_PROMPT.format(headline=headline, body=body)

    response = await client.chat.completions.create(
        model=evaluator_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
        seed=INFERENCE_SEED,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Evaluator returned empty response")
    result = json.loads(content)
    ambivalent_ratio = compute_ambivalent_ratio(body)

    overall = _clamp(float(result["overall_slant_score"]), -1.0, 1.0)
    loaded = _clamp(float(result["loaded_language_score"]), 0.0, 1.0)
    omission = _clamp(float(result["omission_risk_score"]), 0.0, 1.0)
    balance = _clamp(float(result["source_balance_score"]), 0.0, 1.0)
    consensus = _clamp(float(result.get("consensus_preservation_score", 0.5)), 0.0, 1.0)
    attribution = _clamp(float(result.get("attribution_score", 0.5)), 0.0, 1.0)
    confidence = _clamp(float(result["confidence"]), 0.0, 1.0)

    return SlantScore(
        overall_slant_score=overall,
        loaded_language_score=loaded,
        omission_risk_score=omission,
        source_balance_score=balance,
        consensus_preservation_score=consensus,
        attribution_score=attribution,
        ambivalent_word_ratio=ambivalent_ratio,
        confidence=confidence,
        rationale=result["rationale"],
        pass_fail=abs(overall) <= slant_threshold,
    )


def compute_ambivalent_ratio(text: str) -> float:
    """
    Compute the ratio of ambivalent/hedging phrases in the text.
    Hall found that higher ambivalent word usage correlates with
    perceived neutrality.
    """
    text_lower = text.lower()
    word_count = len(text_lower.split())
    if word_count == 0:
        return 0.0

    ambivalent_count = sum(
        len(re.findall(r'\b' + re.escape(marker) + r'\b', text_lower))
        for marker in AMBIVALENT_MARKERS
    )

    return ambivalent_count / word_count


def evaluate_source_balance(
    source_slants: list[float],
) -> dict:
    """
    Check whether the source articles used have balanced political representation.
    Returns a balance report.
    """
    if not source_slants:
        return {"balanced": False, "reason": "no sources"}

    avg_slant = sum(source_slants) / len(source_slants)
    has_left = any(s < -0.15 for s in source_slants)
    has_right = any(s > 0.15 for s in source_slants)
    has_center = any(-0.15 <= s <= 0.15 for s in source_slants)

    return {
        "balanced": has_left and has_right,
        "avg_source_slant": round(avg_slant, 3),
        "has_left": has_left,
        "has_right": has_right,
        "has_center": has_center,
        "source_count": len(source_slants),
    }


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
