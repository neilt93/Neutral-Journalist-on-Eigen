"""
Neutrality calibration loop.

This is the core differentiator: generate -> evaluate -> re-generate until
the article passes the slant threshold. Uses a separate evaluator model
to avoid the self-assessment problem identified by Hall.
"""

from __future__ import annotations

import os

import structlog
from openai import AsyncOpenAI

from src.analysis.slant import evaluate_slant, evaluate_source_balance
from src.generation.writer import generate_article
from src.models import GeneratedArticle, SlantScore, TopicCluster
from src.publishing.gate import check_publish_gate

log = structlog.get_logger()

RECALIBRATION_PROMPT = """The following article was evaluated and found to have issues with neutrality or epistemic accuracy.

## Evaluator scores
- Overall slant: {overall_slant:+.2f} (target: 0.0, threshold: ±0.15)
- Loaded language: {loaded_language:.2f}/1.0 (lower is better)
- Omission risk: {omission_risk:.2f}/1.0 (lower is better)
- Source balance: {source_balance:.2f}/1.0 (higher is better)
- Consensus preservation: {consensus_preservation:.2f}/1.0 (higher is better)
- Attribution: {attribution:.2f}/1.0 (higher is better)
- Ambivalent word ratio: {ambivalent_ratio:.3f} (higher is better)

## Evaluator rationale
{rationale}

## Instructions
Rewrite the article to address the specific issues above:
- If slant is negative, the article leans left - add more right-of-center perspectives
- If slant is positive, the article leans right - add more left-of-center perspectives
- If loaded language is high, replace emotionally charged words with neutral alternatives
- If omission risk is high, add missing countervailing facts or context
- If source balance is low, give more space to underrepresented perspectives
- If consensus preservation is low, do NOT flatten expert consensus into false 50/50 debate - reflect the actual weight of evidence while noting dissent
- If attribution is low, attribute disputed claims to their sources instead of stating them as narrator fact
- Use more hedging/ambivalent language where claims are genuinely contested

Original article:
HEADLINE: {headline}

{body}

Output the revised article in the same format:
HEADLINE: <headline>

<body>"""


async def calibrated_generate(
    cluster: TopicCluster,
    client: AsyncOpenAI,
    generation_model: str,
    evaluator_model: str | None = None,
) -> GeneratedArticle:
    """
    Generate an article and iteratively calibrate it toward neutrality.

    Uses a separate model for evaluation when available (per Hall's finding
    that LLMs can't reliably self-assess bias).
    """
    slant_threshold = float(os.getenv("SLANT_THRESHOLD", "0.15"))
    max_rounds = int(os.getenv("MAX_CALIBRATION_ROUNDS", "3"))

    # Use a different model for evaluation if available, otherwise same model
    eval_model = evaluator_model or generation_model

    # Check source balance first
    source_slants = [a.source_slant for a in cluster.articles]
    balance = evaluate_source_balance(source_slants)
    if not balance["balanced"]:
        log.warning(
            "unbalanced_sources",
            topic=cluster.topic,
            balance=balance,
        )

    # Initial generation
    article = await generate_article(
        topic=cluster.topic,
        sources=cluster.articles,
        client=client,
        model=generation_model,
    )

    # Calibration loop
    for round_num in range(1, max_rounds + 1):
        slant = await evaluate_slant(
            headline=article.headline,
            body=article.body,
            client=client,
            evaluator_model=eval_model,
            slant_threshold=slant_threshold,
        )
        article = article.model_copy(update={
            "slant_score": slant,
            "calibration_rounds": round_num,
        })

        # Record history
        round_record = {
            "round": round_num,
            "slant": slant.overall_slant_score,
            "loaded_language": slant.loaded_language_score,
            "omission_risk": slant.omission_risk_score,
            "source_balance": slant.source_balance_score,
            "consensus_preservation": slant.consensus_preservation_score,
            "attribution": slant.attribution_score,
            "confidence": slant.confidence,
            "rationale": slant.rationale,
        }

        # Use the full multi-dimensional gate to decide pass/fail
        gate = check_publish_gate(article)
        round_record["pass"] = gate.allowed
        round_record["gate_failures"] = gate.failures
        article.calibration_history.append(round_record)

        log.info(
            "calibration_round",
            topic=cluster.topic,
            round=round_num,
            gate_passed=gate.allowed,
            slant=slant.overall_slant_score,
            consensus=slant.consensus_preservation_score,
            attribution=slant.attribution_score,
            failures=gate.failures,
        )

        if gate.allowed:
            log.info("calibration_passed", topic=cluster.topic, rounds=round_num)
            return article

        if round_num == max_rounds:
            log.warning("calibration_incomplete", topic=cluster.topic, failures=gate.failures)
            return article

        # Re-generate with feedback
        article = await _recalibrate(
            article=article,
            slant=slant,
            client=client,
            model=generation_model,
        )

    return article


async def _recalibrate(
    article: GeneratedArticle,
    slant: SlantScore,
    client: AsyncOpenAI,
    model: str,
) -> GeneratedArticle:
    """Ask the model to rewrite an article with neutrality feedback. Returns a new article."""
    import hashlib
    from src.generation.writer import INFERENCE_SEED, SYSTEM_PROMPT, _parse_output

    user_prompt = RECALIBRATION_PROMPT.format(
        overall_slant=slant.overall_slant_score,
        loaded_language=slant.loaded_language_score,
        omission_risk=slant.omission_risk_score,
        source_balance=slant.source_balance_score,
        consensus_preservation=slant.consensus_preservation_score,
        attribution=slant.attribution_score,
        ambivalent_ratio=slant.ambivalent_word_ratio,
        rationale=slant.rationale,
        headline=article.headline,
        body=article.body,
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        seed=INFERENCE_SEED,
    )

    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("Recalibration returned empty response")
    headline, body = _parse_output(raw)

    new_prompt_hash = hashlib.sha256(
        (SYSTEM_PROMPT + user_prompt).encode()
    ).hexdigest()

    return article.model_copy(update={
        "headline": headline,
        "body": body,
        "prompt_hash": new_prompt_hash,
    })
