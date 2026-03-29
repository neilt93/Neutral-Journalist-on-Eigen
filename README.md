# The Neutral Wire

A cryptographically verifiable autonomous journalism agent. Ingests news from politically diverse sources, measures bias using the Andy Hall framework, calibrates articles toward neutrality, and publishes with on-chain attestation proving the editorial pipeline wasn't tampered with.

Built for deployment on [EigenCloud](https://eigencloud.xyz) — the TEE proves the exact code and neutrality checks are what's actually running.

## How it works

```
Ingest → Cluster → Generate → Evaluate → Calibrate → Gate → Attest → Publish
```

1. **Ingest** — Fetches articles from 8 politically diverse sources (NPR, Guardian, Reuters, AP, BBC, WSJ, Economist, Fox News) via RSS and API
2. **Cluster** — Groups articles by topic, enforces balanced sourcing (requires both left and right outlets), filters low-reliability sources, deduplicates syndicated content
3. **Generate** — Writes a neutral article synthesizing all source perspectives via an LLM (OpenAI-compatible, production uses EigenAI)
4. **Evaluate** — A *separate* evaluator model scores the article across 6 dimensions:
   - **Overall slant** (left/right lean)
   - **Loaded language** (emotional/partisan wording)
   - **Omission risk** (missing facts or perspectives)
   - **Source balance** (perspective representation proportional to evidence)
   - **Consensus preservation** (does it flatten scientific consensus into false debate?)
   - **Attribution** (are claims attributed to sources, or laundered into narrator voice?)
5. **Calibrate** — If any dimension fails, feeds back all scores and rationale to the writer for targeted rewriting. Repeats up to 3 rounds.
6. **Gate** — Multi-dimensional publish gate blocks articles that fail *any* threshold (not just slant). A false-balance article with 0.00 slant but 0.3 consensus preservation gets rejected.
7. **Attest** — Publishes a cryptographic receipt on-chain: content hash, source set hash, evaluator output hash, prompt hash, TEE attestation
8. **Publish** — Article goes to the news feed with full transparency metadata

## The trust chain

The verifiability story is about **code integrity**, not inference determinism:

1. Your code (system prompt, calibration logic, evaluator, sourcing rules) enforces neutrality
2. EigenCompute TEE proves that exact Docker image is running unmodified
3. The attestation contract logs per-article receipts on-chain
4. Anyone can verify: "the agent really does run the neutrality pipeline it claims to"

You don't need to prove the LLM gave the same answer twice. You just need to prove you didn't secretly remove the neutrality checks before deploying.

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/neilt93/Neutral-Journalist-on-Eigen.git
cd Neutral-Journalist-on-Eigen
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env: set EIGENAI_API_KEY (OpenAI key works for dev)
# Set EIGENAI_API_BASE=https://api.openai.com/v1 for local dev
# Set GUARDIAN_API_KEY (free from open-platform.theguardian.com)

# 3. Run the web app (autonomous pipeline + news feed UI)
uvicorn src.api:app --port 8000
# Open http://localhost:8000

# 4. Or run as a daemon
python -m src.main
```

## Test suite

```bash
# Unit tests (no LLM needed, ~1 second)
pytest tests/ -k "not GoldSetBehavioral"

# Gold set behavioral tests (requires API key, ~50 seconds)
pytest tests/ -k "GoldSetBehavioral" -v

# Solidity contract tests
cd contracts && npm install && npm test
```

**103 total tests**: 69 unit/mock, 34 gold set behavioral (16 pass on gpt-4o, 18 require frontier model — tagged and auto-skipped).

Gold set covers: false balance (6), asymmetric labeling (5), selective omission (4), subtle framing (4), epistemic accuracy (4), plus obvious bias and edge cases.

## Architecture

```
src/
├── main.py              # Daemon entrypoint (autonomous loop)
├── api.py               # Web app (FastAPI + news feed UI)
├── models.py            # Pydantic models
├── store.py             # Article persistence
├── ingestion/
│   ├── fetcher.py       # RSS + API fetching with image extraction
│   └── parser.py        # Topic clustering, dedup, quality filtering
├── analysis/
│   └── slant.py         # 6-dimension evaluator (Hall framework)
├── generation/
│   ├── writer.py        # Article generation
│   └── calibration.py   # Iterative neutrality calibration loop
├── attestation/
│   └── onchain.py       # Cryptographic attestation + on-chain publishing
├── publishing/
│   ├── gate.py          # Multi-dimensional publish gate
│   └── publisher.py     # AgentKit publishing
└── wallet/
    └── treasury.py      # AgentKit wallet management

contracts/
└── src/AttestationRegistry.sol  # On-chain attestation registry

static/
└── index.html           # News feed frontend ("The Neutral Wire")

tests/
├── test_slant_pipeline.py  # Full test suite
└── gold_set.py             # 34 synthetic bias test cases
```

## EigenCloud deployment

```bash
# Build Docker image
docker build -t neutral-wire .

# Deploy to EigenCompute (TEE)
ecloud compute app deploy --image-ref neutral-wire:latest

# The TEE proves this exact image is running unmodified
# Verify at verify.eigencloud.xyz
```

## Key design decisions

- **Separate evaluator model** — Per Hall's research, LLMs can't reliably self-assess bias. The evaluator must be a different model (or at minimum, a different call) from the generator.
- **Epistemic accuracy over forced symmetry** — The agent optimizes for "neutral tone, accurate epistemics," not equal weight to all sides. When scientific consensus exists, it's reflected proportionally.
- **Multi-dimensional gate** — Slant alone isn't enough. A false-balance article can score 0.00 slant while being deeply misleading. The gate checks all 6 dimensions.
- **Source diversity enforcement** — Articles are only generated when sources from both left and right-leaning outlets are available. Center-only clusters are rejected when balanced sourcing is required.

## References

- [Andy Hall — Measuring Perceived Slant in LLMs](https://www.hoover.org/research/measuring-perceived-slant-large-language-models-through-user-evaluations) — The framework for slant evaluation
- [EigenCloud — Verifiable Agents on EigenLayer](https://blog.eigencloud.xyz/introducing-verifiable-agents-on-eigenlayer/) — TEE infrastructure
- [Sovra](https://mpost.io/eigencloud-introduces-agentkit-to-power-verifiable-revenue-generating-ai-agents-with-onchain-capabilities/) — EigenCloud's reference autonomous agent
