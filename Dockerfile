# Multi-stage Dockerfile for all dataset-viewer services and jobs
# Build with: docker build --target <service_name> -t <tag> .

ARG PYTHON_VERSION=3.12.11
FROM python:${PYTHON_VERSION}-slim AS viewer

# Install Rust and minimal build deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain and maturin
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && . $HOME/.cargo/env \
    && pip install maturin \
    && rustc --version \
    && cargo --version
# Add cargo bin dir to PATH (so maturin + cargo available globally)
ENV PATH="/root/.cargo/bin:${PATH}"

# Build libviewer
COPY libs/libviewer /src/libs/libviewer
WORKDIR /src/libs/libviewer
RUN maturin build --release --strip --out /tmp/dist

# Base stage with shared setup
FROM python:${PYTHON_VERSION}-slim AS libcommon

# System dependencies
RUN apt-get update \
    && apt-get install -y unzip wget procps htop ffmpeg libavcodec-extra libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Common environment variables
ARG POETRY_VERSION=2.1.4
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PATH="$PATH:/root/.local/bin"

# Install pip and poetry
RUN pip install -U pip && pip install "poetry==${POETRY_VERSION}"

# Add dummy pyproject.toml for libviewer so that poetry install
# can resolve it but we later overwrite libviewer with the prebuilt
# wheel.
RUN mkdir -p /src/libs/libviewer && \
    printf "[project]\nname = \"libviewer\"\nversion = \"0.0.0\"\n" > /src/libs/libviewer/pyproject.toml

# Install libcommon's dependencies but not libcommon itself.
COPY libs/libcommon/poetry.lock \
     libs/libcommon/pyproject.toml \
     /src/libs/libcommon/
WORKDIR /src/libs/libcommon
RUN poetry install --no-cache --no-root

# Install libviewer wheel built in the viewer stage
COPY --from=viewer /tmp/dist /tmp/dist
RUN pip install /tmp/dist/libviewer-*.whl

# Add libcommon source but do not install
COPY libs/libcommon /src/libs/libcommon
RUN poetry install --no-cache --only-root

# Base image for all services
FROM libcommon AS libapi
COPY libs/libapi /src/libs/libapi
WORKDIR /src/libs/libapi
RUN poetry install --no-cache --only-root

# Below are the actual API services which depend on libapi and libcommon.
# Since the majority of the dependencies are already installed in the
# `common` stage we let poetry to handle the rest.

# API service
FROM libapi AS api
COPY services/api /src/services/api
WORKDIR /src/services/api
RUN poetry install --no-cache --only-root
CMD ["poetry", "run", "python", "src/api/main.py"]

# Admin service
FROM libapi AS admin
COPY services/admin /src/services/admin
WORKDIR /src/services/admin
RUN poetry install --no-cache
CMD ["poetry", "run", "python", "src/admin/main.py"]

# Rows service
FROM libapi AS rows
COPY services/rows /src/services/rows
WORKDIR /src/services/rows
RUN poetry install --no-cache --only-root
CMD ["poetry", "run", "python", "src/rows/main.py"]

# Search service
FROM libapi AS search
COPY services/search /src/services/search
WORKDIR /src/services/search
RUN poetry install --no-cache --only-root
CMD ["poetry", "run", "python", "src/search/main.py"]

# SSE API service
FROM libapi AS sse-api
COPY services/sse-api /src/services/sse-api
WORKDIR /src/services/sse-api
RUN poetry install --no-cache --only-root
CMD ["poetry", "run", "python", "src/sse_api/main.py"]

# Webhook service
FROM libapi AS webhook
COPY services/webhook /src/services/webhook
WORKDIR /src/services/webhook
RUN poetry install --no-cache --only-root
CMD ["poetry", "run", "python", "src/webhook/main.py"]

# Worker service
FROM libcommon AS worker
COPY services/worker /src/services/worker
WORKDIR /src/services/worker
# presidio-analyzer > spacy > thinc doesn't ship aarch64 wheels so need to compile
RUN if [ "$(uname -m)" = "aarch64" ]; then \
      apt-get update && apt-get install -y build-essential && \
      rm -rf /var/lib/apt/lists/*; \
    fi
RUN poetry install --no-cache --only-root
RUN python -m spacy download en_core_web_lg
CMD ["poetry", "run", "python", "src/worker/main.py"]

# Cache maintenance job
FROM libcommon AS cache_maintenance
COPY jobs/cache_maintenance /src/jobs/cache_maintenance
WORKDIR /src/jobs/cache_maintenance
RUN poetry install --no-cache --only-root
CMD ["poetry", "run", "python", "src/cache_maintenance/main.py"]

# MongoDB migration job
FROM libcommon AS mongodb_migration
COPY jobs/mongodb_migration /src/jobs/mongodb_migration
WORKDIR /src/jobs/mongodb_migration
RUN poetry install --no-cache --only-root
CMD ["poetry", "run", "python", "src/mongodb_migration/main.py"]
