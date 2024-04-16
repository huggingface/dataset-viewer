# build with
#   docker build -t some_tag_worker -f Dockerfile ../..
FROM python:3.9.18-slim

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    # Versions:
    POETRY_VERSION=1.8.2 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PATH="$PATH:/root/.local/bin"

# System deps:
RUN apt-get update \
    && apt-get install -y unzip wget procps htop ffmpeg libavcodec-extra libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip
RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /src
COPY services/worker/poetry.lock ./services/worker/poetry.lock
COPY services/worker/pyproject.toml ./services/worker/pyproject.toml
COPY libs/libcommon ./libs/libcommon

WORKDIR /src/services/worker/
RUN poetry install --no-cache
COPY services/worker/src ./src
RUN poetry install --no-cache

ENTRYPOINT ["poetry", "run", "python", "src/worker/main.py"]
