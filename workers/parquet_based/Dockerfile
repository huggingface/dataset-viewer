# build with
#   docker build -t some_tag_worker -f Dockerfile ../..
FROM python:3.9.15-slim

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    # Versions:
    POETRY_VERSION=1.2.2 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

# # System deps:
# RUN apt-get update \
#     && apt-get install -y build-essential unzip wget python3-dev pkg-config \
#     && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir pip
RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /src
COPY libs/libcommon/dist ./libs/libcommon/dist
COPY workers/parquet_based/src ./workers/parquet_based/src
COPY workers/parquet_based/poetry.lock ./workers/parquet_based/poetry.lock
COPY workers/parquet_based/pyproject.toml ./workers/parquet_based/pyproject.toml
WORKDIR /src/workers/parquet_based/
RUN poetry install

ENTRYPOINT ["poetry", "run", "python", "src/parquet_based/main.py"]
