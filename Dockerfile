# PHAROS — reproducible container image.
#
# Multi-stage so the runtime image carries neither the build toolchain nor the
# package index cache. The result runs the whole pipeline -- fetch, clean,
# index, evaluate, query -- with no API key and no model download, because the
# default encoder is fitted on the corpus itself.

# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependency metadata first, so a source-only change does not invalidate the
# dependency layer.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install .

# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="PHAROS" \
      org.opencontainers.image.description="Cohort-grounded retrieval-augmented synthesis over patient drug reviews" \
      org.opencontainers.image.licenses="MIT"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY configs/ ./configs/
COPY docs/ ./docs/
COPY README.md LICENSE ./

# Data and artifacts are mounted, not baked: the corpus is 110 MB and licensed
# to its publisher, and an image that ships it would be both large and wrong.
RUN mkdir -p data/raw data/processed data/index artifacts \
    && useradd --create-home --uid 1000 pharos \
    && chown -R pharos:pharos /app
USER pharos

VOLUME ["/app/data", "/app/artifacts"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pharos info || exit 1

ENTRYPOINT ["pharos"]
CMD ["--help"]

# Build:  docker build -t pharos-rx .
# Set up: docker run --rm -v pharos-data:/app/data pharos-rx fetch-data
#         docker run --rm -v pharos-data:/app/data pharos-rx build-corpus
#         docker run --rm -v pharos-data:/app/data pharos-rx build-index
# Ask:    docker run --rm -v pharos-data:/app/data pharos-rx \
#           ask "What side effects do reviewers report on metformin?"
