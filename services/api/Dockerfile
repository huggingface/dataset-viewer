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
    POETRY_VERSION=1.3.2 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PATH="$PATH:/root/.local/bin"

# System deps:
RUN apt-get update \
    && apt-get install -y build-essential unzip wget \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -U pip
RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /src
COPY services/api/poetry.lock ./services/api/poetry.lock
COPY services/api/pyproject.toml ./services/api/pyproject.toml
COPY libs/libcommon ./libs/libcommon
WORKDIR /src/services/api/
RUN poetry install --no-cache
COPY services/api/src ./src
RUN poetry install --no-cache

ENTRYPOINT ["poetry", "run", "python", "src/api/main.py"]
