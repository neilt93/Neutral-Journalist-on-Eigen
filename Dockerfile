FROM python:3.12-slim

# EigenCompute TEE metadata
LABEL eigen.compute.tee="true"
LABEL eigen.compute.attestation="enabled"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY config/ config/
COPY src/ src/

# Non-root user for security
RUN useradd --create-home agent
USER agent

# Health check for EigenCompute monitoring
HEALTHCHECK --interval=60s --timeout=10s \
    CMD python -c "import os, sys, time; from pathlib import Path; path = Path(os.getenv('AGENT_HEARTBEAT_PATH', '/tmp/neutral_journalism_agent.heartbeat')); max_age = (int(os.getenv('AGENT_LOOP_INTERVAL_MINUTES', '60')) + 15) * 60; sys.exit(0 if path.exists() and time.time() - path.stat().st_mtime <= max_age else 1)"

ENTRYPOINT ["python", "-m", "src.main"]
