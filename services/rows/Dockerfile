# build with
#   docker build -t some_tag_rows -f Dockerfile ../..
FROM python:3.9.15-slim

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    # Versions:
    POETRY_VERSION=1.4.2 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PATH="$PATH:/root/.local/bin"

# System deps:
RUN apt-get update \
    && apt-get install -y build-essential unzip wget \
    libicu-dev ffmpeg libavcodec-extra libsndfile1 llvm pkg-config \
    procps htop \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -U pip
RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /src
COPY services/rows/poetry.lock ./services/rows/poetry.lock
COPY services/rows/pyproject.toml ./services/rows/pyproject.toml
COPY libs/libcommon ./libs/libcommon
COPY libs/libapi ./libs/libapi
WORKDIR /src/services/rows/
RUN poetry install --no-cache
COPY services/rows/src ./src
RUN poetry install --no-cache

ENTRYPOINT ["poetry", "run", "python", "src/rows/main.py"]
