"""
Onchain attestation for article provenance.

Every published article gets a cryptographic attestation proving:
1. Which source articles were used (source set hash)
2. What model produced the output and how it scored (evaluator output hash)
3. What prompts were used (prompt config hash)
4. The slant score and calibration rounds
5. TEE attestation from EigenCompute proving the code wasn't modified

This is published to the AttestationRegistry contract so anyone can
verify the agent's editorial process.
"""

from __future__ import annotations

import hashlib
import os

import structlog
from eth_account import Account
from web3 import AsyncWeb3

from src.models import AttestationRecord, GeneratedArticle

log = structlog.get_logger()

# ABI matching contracts/src/AttestationRegistry.sol
ATTESTATION_ABI = [
    {
        "inputs": [
            {"name": "articleHash", "type": "bytes32"},
            {"name": "sourceSetHash", "type": "bytes32"},
            {"name": "evaluatorOutputHash", "type": "bytes32"},
            {"name": "promptConfigHash", "type": "bytes32"},
            {"name": "slantScore", "type": "int16"},
            {"name": "calibrationRounds", "type": "uint8"},
            {"name": "teeAttestation", "type": "bytes"},
        ],
        "name": "publish",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"name": "articleHash", "type": "bytes32"}],
        "name": "lookup",
        "outputs": [
            {"name": "found", "type": "bool"},
            {"name": "id", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "count",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]


def _compute_source_set_hash(source_hashes: list[str]) -> str:
    """SHA-256 of sorted source content hashes, for deterministic comparison."""
    joined = ",".join(sorted(source_hashes))
    return hashlib.sha256(joined.encode()).hexdigest()


def _compute_evaluator_output_hash(article: GeneratedArticle) -> str:
    """SHA-256 of the full SlantScore JSON, so the exact evaluation is attestable."""
    if article.slant_score is None:
        return hashlib.sha256(b"no-evaluation").hexdigest()
    return hashlib.sha256(
        article.slant_score.model_dump_json().encode()
    ).hexdigest()


async def create_attestation(
    article: GeneratedArticle,
    tee_attestation: str,
) -> AttestationRecord:
    """Build an attestation record for a generated article."""
    source_hashes = [a.content_hash for a in article.sources_used]
    return AttestationRecord(
        article_hash=article.content_hash,
        source_set_hash=_compute_source_set_hash(source_hashes),
        evaluator_output_hash=_compute_evaluator_output_hash(article),
        prompt_config_hash=article.prompt_hash,
        model_id=article.model_id,
        slant_score=article.slant_score.overall_slant_score if article.slant_score else 0.0,
        calibration_rounds=article.calibration_rounds,
        tee_attestation=tee_attestation,
    )


async def publish_onchain(record: AttestationRecord) -> str | None:
    """
    Publish an attestation record to the AttestationRegistry contract.
    Returns the transaction hash, or None if not configured.
    """
    rpc_url = os.getenv("ATTESTATION_RPC_URL")
    contract_addr = os.getenv("ATTESTATION_CONTRACT_ADDRESS")
    private_key = os.getenv("AGENT_WALLET_PRIVATE_KEY")

    if not all([rpc_url, contract_addr, private_key]):
        log.warning("attestation_not_configured", missing="check env vars")
        _log_attestation(record)
        return None

    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
    contract = w3.eth.contract(
        address=w3.to_checksum_address(contract_addr),
        abi=ATTESTATION_ABI,
    )
    account = Account.from_key(private_key)
    del private_key  # Don't keep raw key in local scope (traceback exposure risk)

    # Convert hex strings to bytes32
    article_hash = bytes.fromhex(record.article_hash)
    source_set_hash = bytes.fromhex(record.source_set_hash)
    evaluator_hash = bytes.fromhex(record.evaluator_output_hash)
    prompt_hash = bytes.fromhex(record.prompt_config_hash)
    slant_scaled = int(record.slant_score * 1000)  # float → int16
    tee_bytes = bytes.fromhex(record.tee_attestation) if record.tee_attestation else b""

    max_gas = int(os.getenv("ATTESTATION_MAX_GAS", "500000"))
    tx = await contract.functions.publish(
        article_hash,
        source_set_hash,
        evaluator_hash,
        prompt_hash,
        slant_scaled,
        record.calibration_rounds,
        tee_bytes,
    ).build_transaction({
        "from": account.address,
        "nonce": await w3.eth.get_transaction_count(account.address),
        "gas": max_gas,
    })

    signed = account.sign_transaction(tx)
    tx_hash = await w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = await w3.eth.wait_for_transaction_receipt(tx_hash)

    record.tx_hash = receipt["transactionHash"].hex()
    log.info("attestation_published", tx_hash=record.tx_hash)
    return record.tx_hash


def _log_attestation(record: AttestationRecord) -> None:
    """Log attestation locally when onchain publishing isn't configured."""
    log.info(
        "attestation_local",
        article_hash=record.article_hash[:16] + "...",
        source_set_hash=record.source_set_hash[:16] + "...",
        evaluator_hash=record.evaluator_output_hash[:16] + "...",
        model=record.model_id,
        slant=record.slant_score,
        rounds=record.calibration_rounds,
    )


async def get_tee_attestation() -> str:
    """
    Get TEE attestation from the EigenCompute runtime.

    Inside an EigenCompute TEE, the container launcher exposes a Unix socket
    at /run/container_launcher/teeserver.sock. We POST a challenge to
    /v1/bound_evidence and receive Intel TDX hardware attestation bytes
    proving the running Docker image matches the whitelisted digest.

    Outside a TEE (local dev), returns a deterministic placeholder hash.
    """
    tee_socket = "/run/container_launcher/teeserver.sock"
    # Fall back to env var for custom setups, then check for the socket
    endpoint = os.getenv("EIGENCOMPUTE_TEE_ATTESTATION_ENDPOINT")

    import httpx
    from pathlib import Path

    if endpoint:
        # Custom endpoint (e.g. remote KMS proxy)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(endpoint, json={"challenge": os.urandom(32).hex()})
            resp.raise_for_status()
            return resp.json()["attestation"]

    if Path(tee_socket).exists():
        # Inside EigenCompute TEE — use the local Unix socket
        transport = httpx.AsyncHTTPTransport(uds=tee_socket)
        async with httpx.AsyncClient(transport=transport, timeout=30) as client:
            challenge = os.urandom(32).hex()
            resp = await client.post(
                "http://localhost/v1/bound_evidence",
                json={"challenge": challenge},
            )
            resp.raise_for_status()
            return resp.json()["attestation"]

    log.info("tee_not_available", msg="running outside TEE, using placeholder")
    return hashlib.sha256(b"dev-tee-attestation").hexdigest()
