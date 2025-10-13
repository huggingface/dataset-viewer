# Multi-stage Dockerfile for all dataset-viewer services and jobs
# Build with: docker build --target <service_name> -t <tag> .

# Base stage with shared setup
FROM python:3.12.11-slim AS common

# System dependencies
RUN apt-get update \
    && apt-get install -y unzip wget procps htop ffmpeg libavcodec-extra libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Common environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VERSION=2.1.4 \
    POETRY_VIRTUALENVS_CREATE=false \
    PATH="$PATH:/root/.local/bin"

# Install pip and poetry
RUN pip install -U pip && pip install "poetry==$POETRY_VERSION"

# Install libcommon's dependencies but not libcommon itself
COPY libs/libcommon/poetry.lock \
     libs/libcommon/pyproject.toml \
     /src/libs/libcommon/
RUN poetry install --no-cache --no-root --no-directory -P /src/libs/libcommon

# Base image for services including libapi's dependencies
FROM common AS api
COPY libs/libapi/poetry.lock \
     libs/libapi/pyproject.toml \
     /src/libs/libapi/
RUN poetry install --no-cache --no-root --no-directory -P /src/libs/libapi

# Below are the actual API services which depend on libapi and libcommon.
# Since the majority of the dependencies are already installed in the `api`
# we let poetry to actually install the `libs` and the specific service.

# Admin service
FROM api AS admin
COPY libs /src/libs
COPY services/admin /src/services/admin
RUN poetry install --no-cache -P /src/services/admin
WORKDIR /src/services/admin/
ENTRYPOINT ["poetry", "run", "python", "src/admin/main.py"]

# Rows service
FROM api AS rows
COPY libs /src/libs
COPY services/rows /src/services/rows
RUN poetry install --no-cache -P /src/services/rows
WORKDIR /src/services/rows/
ENTRYPOINT ["poetry", "run", "python", "src/rows/main.py"]

# Search service
FROM api AS search
COPY libs /src/libs
COPY services/search /src/services/search
RUN poetry install --no-cache -P /src/services/search
WORKDIR /src/services/search/
ENTRYPOINT ["poetry", "run", "python", "src/search/main.py"]

# SSE API service
FROM api AS sse-api
COPY libs /src/libs
COPY services/sse-api /src/services/sse-api
RUN poetry install --no-cache -P /src/services/sse-api
WORKDIR /src/services/sse-api/
ENTRYPOINT ["poetry", "run", "python", "src/sse_api/main.py"]

# Webhook service
FROM api AS webhook
COPY libs /src/libs
COPY services/webhook /src/services/webhook
RUN poetry install --no-cache -P /src/services/webhook
WORKDIR /src/services/webhook/
ENTRYPOINT ["poetry", "run", "python", "src/webhook/main.py"]

# Worker
FROM common AS worker
RUN poetry run python -m spacy download en_core_web_lg
COPY libs /src/libs
COPY services/worker /src/services/worker
RUN poetry install --no-cache -P /src/services/worker
WORKDIR /src/services/worker/
ENTRYPOINT ["poetry", "run", "python", "src/worker/main.py"]

# Cache maintenance job
FROM common AS cache-maintenance
COPY libs /src/libs
COPY jobs/cache_maintenance /src/jobs/cache_maintenance
RUN poetry install --no-cache -P /src/jobs/cache_maintenance
WORKDIR /src/jobs/cache_maintenance/
ENTRYPOINT ["poetry", "run", "python", "src/cache_maintenance/main.py"]

# MongoDB migration job
FROM common AS mongodb-migration
COPY libs /src/libs
COPY jobs/mongodb_migration /src/jobs/mongodb_migration
RUN poetry install --no-cache -P /src/jobs/mongodb_migration
WORKDIR /src/jobs/mongodb_migration/
ENTRYPOINT ["poetry", "run", "python", "src/data_migration/main.py"]