"""
Agent treasury management via AgentKit.

The agent is economically autonomous — it has its own wallet, can pay for
compute (EigenAI inference, EigenCompute), and manage funds. AgentKit
provides the wallet primitives.
"""

from __future__ import annotations

import os

import httpx
import structlog

log = structlog.get_logger()


async def get_balance() -> dict:
    """Get the agent's current treasury balance."""
    api_base = os.getenv("AGENTKIT_API_BASE", "https://agentkit.eigen.ai")
    api_key = os.getenv("AGENTKIT_API_KEY")

    if not api_key:
        return {"status": "not_configured"}

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{api_base}/v1/wallet/balance",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        return resp.json()


async def check_funds_sufficient(estimated_cost_usd: float) -> bool:
    """Check if the agent has enough funds for an operation."""
    balance = await get_balance()
    if balance.get("status") == "not_configured":
        log.warning("wallet_not_configured")
        return True  # Allow operation in dev mode

    available = balance.get("available_usd", 0.0)
    if available < estimated_cost_usd:
        log.warning(
            "insufficient_funds",
            available=available,
            required=estimated_cost_usd,
        )
        return False
    return True


async def estimate_cycle_cost(
    num_sources: int,
    max_calibration_rounds: int,
) -> float:
    """
    Estimate the cost of one agent cycle (ingest → generate → calibrate → publish).
    Rough estimates based on typical LLM pricing.
    """
    # Rough token estimates
    ingestion_tokens = num_sources * 2000  # ~2K tokens per source article
    generation_tokens = 3000  # Article generation
    evaluation_tokens = 2000 * max_calibration_rounds  # Slant eval per round
    recalibration_tokens = 3000 * (max_calibration_rounds - 1)  # Re-gen rounds
    attestation_gas = 0.50  # Estimated gas cost in USD

    total_tokens = (
        ingestion_tokens + generation_tokens +
        evaluation_tokens + recalibration_tokens
    )
    # Rough cost: $0.01 per 1K tokens (varies by model)
    inference_cost = (total_tokens / 1000) * 0.01
    return inference_cost + attestation_gas
