# The Neutral Wire

A cryptographically verifiable autonomous journalism pipeline and news feed. It ingests news from politically diverse sources, scores bias and epistemic quality using an Andy Hall-inspired framework, calibrates drafts toward neutrality, and can publish with on-chain attestation proving the editorial pipeline was not tampered with.

Built for deployment on [EigenCloud](https://eigencloud.xyz). In the daemon/production path, an EigenCompute TEE proves the exact Docker image and neutrality checks are what actually ran.

## How it works

```text
Ingest -> Cluster -> Generate -> Evaluate -> Calibrate -> Gate -> Attest -> Publish
```

1. **Ingest** - Fetches articles from 8 politically diverse sources (NPR, The Guardian, Reuters, Associated Press, BBC News, Wall Street Journal, The Economist, Fox News) via RSS and API.
2. **Cluster** - Groups articles by topic, enforces minimum source coverage, optionally requires both left and right outlets, filters low-reliability sources, and deduplicates syndicated content.
3. **Generate** - Writes a neutral article synthesizing source perspectives with an OpenAI-compatible LLM. Production deployments can point this at EigenAI.
4. **Evaluate** - A separate evaluator call scores the article across 6 dimensions. When `EIGENAI_EVALUATOR_MODEL` is set, evaluation can use a different model; otherwise it falls back to the generation model on a separate call.
   - **Overall slant**: left/right lean
   - **Loaded language**: emotional or partisan wording
   - **Omission risk**: missing facts or perspectives
   - **Source balance**: whether representation is proportional to the evidence
   - **Consensus preservation**: whether the article preserves real expert consensus instead of flattening it into false debate
   - **Attribution**: whether disputed claims stay tied to their sources instead of being laundered into narrator voice
5. **Calibrate** - If the multi-dimensional gate fails, the evaluator's scores and rationale are fed back to the writer for targeted rewriting. The loop repeats up to 3 rounds.
6. **Gate** - Blocks publication on any failing threshold, including slant, loaded language, omission risk, source balance, consensus preservation, attribution, and optionally missing on-chain attestation.
7. **Attest** - Builds a receipt containing the article hash, source-set hash, evaluator-output hash, prompt hash, calibration rounds, and TEE attestation. The daemon path publishes this on-chain when configured.
8. **Publish** - The daemon path publishes externally via AgentKit once required attestation exists. The web app path stores articles locally and serves them in the feed UI with transparency metadata.

## The trust chain

The verifiability story is about **code integrity**, not proving that an LLM always returns the same string:

1. Your code enforces neutrality through prompts, calibration logic, evaluator calls, sourcing rules, and publish gating.
2. EigenCompute TEE proves that exact Docker image is running unmodified.
3. The attestation contract logs per-article receipts on-chain.
4. Anyone can verify that the deployed agent really ran the neutrality pipeline it claims to run.

You do not need to prove the model produced the same output twice. You need to prove no one secretly removed the neutrality checks before deployment.

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/neilt93/Neutral-Journalist-on-Eigen.git
cd Neutral-Journalist-on-Eigen
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env: set EIGENAI_API_KEY (an OpenAI key works for local dev)
# For local dev, set EIGENAI_API_BASE=https://api.openai.com/v1
# Set GUARDIAN_API_KEY (free from open-platform.theguardian.com)

# 3. Run the web app (background pipeline + news feed UI)
uvicorn src.api:app --port 8000
# Open http://localhost:8000

# 4. Or run as a daemon
python -m src.main
```

## Test suite

```bash
# Non-behavioral Python tests (no live evaluator required)
pytest tests/ -k "not GoldSetBehavioral"

# Gold-set behavioral tests (requires API key)
pytest tests/ -k "GoldSetBehavioral" -v

# Solidity contract tests
cd contracts && npm install && npm test
```

**Python tests collected:** 107

**Current local status (March 29, 2026):**
- `pytest -q`: 86 passed, 3 failed, 18 skipped
- `npm test` in `contracts/`: 11 passing

The 3 remaining behavioral failures on `gpt-4o` are:
- `left_moderate_immigration`
- `subtle_emotional_verbs`
- `subtle_selective_omission`

The gold set contains 34 synthetic behavioral cases. In the current `gpt-4o` run, 13 pass, 3 fail, and 18 frontier-tagged cases skip.

Gold-set coverage includes false balance (6), asymmetric labeling (5), selective omission (5), subtle framing (5), epistemic accuracy (4), plus more obvious bias and edge cases.

## Architecture

```text
src/
|-- main.py                 # Daemon entrypoint (autonomous loop)
|-- api.py                  # FastAPI web app for The Neutral Wire
|-- models.py               # Pydantic models
|-- article_selection.py    # Prompt-sized balanced article sampling
|-- store.py                # Article persistence and pipeline log
|-- ingestion/
|   |-- fetcher.py          # RSS + API fetching with text/image extraction
|   `-- parser.py           # Topic clustering, dedup, quality filtering
|-- analysis/
|   `-- slant.py            # 6-dimension evaluator
|-- generation/
|   |-- writer.py           # Article generation
|   `-- calibration.py      # Iterative neutrality calibration loop
|-- attestation/
|   `-- onchain.py          # Cryptographic attestation + on-chain publishing
|-- publishing/
|   |-- gate.py             # Multi-dimensional publish gate
|   `-- publisher.py        # AgentKit publishing
`-- wallet/
    `-- treasury.py         # AgentKit wallet management

contracts/
`-- src/AttestationRegistry.sol  # On-chain attestation registry

static/
`-- index.html              # The Neutral Wire frontend

tests/
|-- test_slant_pipeline.py  # Python test suite
`-- gold_set.py             # 34 synthetic bias test cases
```

## EigenCloud deployment

```bash
# Build Docker image
docker build -t neutral-wire .

# Deploy to EigenCompute (TEE)
ecloud compute app deploy --image-ref neutral-wire:latest
```

The TEE proves that exact image is running unmodified.

## Key design decisions

- **Separate evaluator call** - Per Hall's research, LLMs are poor at self-assessing bias. A separate evaluator model is supported and preferred, but the repo can also fall back to the generation model on a separate evaluation call.
- **Epistemic accuracy over forced symmetry** - The goal is neutral tone with accurate epistemics, not equal weight to all sides. When expert consensus exists, the system aims to reflect it proportionally.
- **Multi-dimensional gate** - Slant alone is not enough. A false-balance article can score near `0.00` on slant while still being misleading, so the gate checks multiple dimensions before publication.
- **Source diversity enforcement** - Articles are only generated when enough distinct sources are present, and balanced-sourcing mode requires both left and right-leaning outlets.
- **Attestation-aware publishing** - In the daemon path, external publish can be blocked unless the article's attestation lands on-chain.

## References

- [Andy Hall - Measuring Perceived Slant in LLMs](https://www.hoover.org/research/measuring-perceived-slant-large-language-models-through-user-evaluations)
- [EigenCloud - Verifiable Agents on EigenLayer](https://blog.eigencloud.xyz/introducing-verifiable-agents-on-eigenlayer/)
- [Sovra](https://mpost.io/eigencloud-introduces-agentkit-to-power-verifiable-revenue-generating-ai-agents-with-onchain-capabilities/)
