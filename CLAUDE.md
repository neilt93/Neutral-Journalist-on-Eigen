# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

The Neutral Wire — an autonomous journalism pipeline on EigenCloud. Ingests news from politically diverse sources, evaluates bias across multiple dimensions, iteratively calibrates articles toward neutrality, and publishes with cryptographic on-chain attestation proving editorial integrity via TEE.

## Commands

```bash
# Install
pip install -r requirements.txt
cp .env.example .env  # then fill in API keys

# Run web app (FastAPI + background pipeline)
uvicorn src.api:app --port 8000

# Run daemon (autonomous production loop)
python -m src.main

# Docker
docker-compose up web    # web UI
docker-compose up agent  # daemon

# Tests
pytest tests/ -k "not GoldSetBehavioral"              # unit tests (no API keys needed)
pytest tests/ -k "GoldSetBehavioral" -v               # behavioral gold-set tests (needs API key)
pytest tests/test_slant_pipeline.py::TestClassName::test_name  # single test

# Solidity contracts
cd contracts && npm install && npm test
```

## Architecture

**Pipeline flow:** Ingest → Cluster → Generate → Evaluate → Calibrate → Gate → Attest → Publish

- **Ingestion** (`src/ingestion/fetcher.py`): RSS + API fetching from 8 sources with `newspaper4k` extraction
- **Clustering** (`src/ingestion/parser.py`): Groups articles by topic, deduplicates, enforces minimum source coverage (3+)
- **Generation** (`src/generation/writer.py`): Synthesizes neutral article from cluster via OpenAI-compatible LLM
- **Evaluation** (`src/analysis/slant.py`): Separate evaluator model scores 6 dimensions (slant, loaded language, omission risk, source balance, consensus preservation, attribution). Uses a *different* model than the writer — per Andy Hall's research that LLMs are poor at self-assessing bias
- **Calibration** (`src/generation/calibration.py`): If evaluation fails, feeds scores back to writer for rewrite (up to 3 rounds)
- **Gate** (`src/publishing/gate.py`): Multi-dimensional publish check — blocks if ANY dimension fails threshold
- **Attestation** (`src/attestation/onchain.py`): SHA-256 hashes of article, sources, evaluator output, and prompts → on-chain via `AttestationRegistry.sol`
- **Publishing** (`src/publishing/publisher.py`): AgentKit social publishing + local JSON store

**Smart contract** (`contracts/src/AttestationRegistry.sol`): Immutable append-only registry. Only registered agent can publish. Stores article hash, source set hash, evaluator output hash, prompt config hash, slant score, calibration rounds, TEE attestation.

## Key Design Decisions

- **Separate evaluator model**: Generation model != evaluation model (configurable via `EIGENAI_EVALUATOR_MODEL`)
- **Deterministic inference**: `EIGENAI_INFERENCE_SEED=42` ensures reproducibility for cryptographic verification
- **No database**: JSON file store (`data/articles.json`) for simplicity and verifiability
- **Epistemic accuracy over forced symmetry**: Goal is neutral tone with accurate epistemics, not equal weight to all positions
- **Multi-dimensional gate**: Slant score alone is insufficient; loaded language, omission risk, source balance, consensus preservation, and attribution are all checked independently

## Configuration

- `config/sources.yaml`: 8 news sources with editorial slant estimates and reliability ratings
- `config/topics.yaml`: 10 topic categories with keyword matching and priority levels
- `.env`: All thresholds (`SLANT_THRESHOLD`, `SOURCE_BALANCE_THRESHOLD`, `MAX_CALIBRATION_ROUNDS`) and API configuration

## Data Models

All Pydantic models in `src/models.py`: `SourceConfig`, `IngestedArticle`, `TopicCluster`, `SlantScore`, `GeneratedArticle`, `AttestationRecord`.
