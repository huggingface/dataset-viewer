# build with
#   docker build -t some_tag_worker -f Dockerfile ../..
FROM python:3.9.6-slim

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    # Versions:
    POETRY_VERSION=1.1.12 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

# System deps:
RUN apt-get update \
    && apt-get install -y build-essential unzip wget python3-dev make \
    libicu-dev ffmpeg libavcodec-extra libsndfile1 llvm pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir pip
RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /src
COPY libs/libcache/dist ./libs/libcache/dist
COPY libs/libqueue/dist ./libs/libqueue/dist
COPY libs/libcommon/dist ./libs/libcommon/dist
COPY workers/splits/src ./workers/splits/src
COPY workers/splits/poetry.lock ./workers/splits/poetry.lock
COPY workers/splits/pyproject.toml ./workers/splits/pyproject.toml
COPY vendors ./vendors/
WORKDIR /src/workers/splits/
RUN poetry install

ENTRYPOINT ["poetry", "run", "python", "src/splits/main.py"]
