# build with
#   docker build -t some_tag_worker -f Dockerfile ../..
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
    && apt-get install -y build-essential unzip wget make \
    libicu-dev ffmpeg libavcodec-extra libsndfile1 llvm pkg-config \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip
RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /src
COPY services/worker/vendors ./services/worker/vendors/
COPY services/worker/poetry.lock ./services/worker/poetry.lock
COPY services/worker/pyproject.toml ./services/worker/pyproject.toml
COPY libs/libcommon ./libs/libcommon
WORKDIR /src/services/worker/
RUN poetry install --no-cache
COPY services/worker/src ./src
RUN poetry install --no-cache

ENTRYPOINT ["poetry", "run", "python", "src/worker/main.py"]
