# build with
#   docker build -t some_tag_api -f Dockerfile ../..
FROM python:3.9.15-slim

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    # Versions:
    POETRY_VERSION=1.4.0 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PATH="$PATH:/root/.local/bin"

# System deps:
RUN apt-get update \
    && apt-get install -y build-essential unzip wget \
    libicu-dev ffmpeg libavcodec-extra libsndfile1 llvm pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -U pip
RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /src
COPY libs/libcommon/poetry.lock ./libs/libcommon/poetry.lock
COPY libs/libcommon/pyproject.toml ./libs/libcommon/pyproject.toml
COPY services/api/poetry.lock ./services/api/poetry.lock
COPY services/api/pyproject.toml ./services/api/pyproject.toml

# FOR LOCAL DEVELOPMENT ENVIRONMENT
# Initialize an empty libcommon
# Mapping a volume to ./libs/libcommon/src is required when running this image.
RUN mkdir ./libs/libcommon/src && mkdir ./libs/libcommon/src/libcommon && touch ./libs/libcommon/src/libcommon/__init__.py

# Install dependencies
WORKDIR /src/services/api/
RUN --mount=type=cache,target=/home/.cache/pypoetry/cache \
    --mount=type=cache,target=/home/.cache/pypoetry/artifacts \
    poetry install --no-root

# FOR LOCAL DEVELOPMENT ENVIRONMENT
# Install the api package.
# Mapping a volume to ./services/api/src is required when running this image.
ENTRYPOINT ["/bin/sh", "-c" , "poetry install --only-root && poetry run python src/api/main.py"]
