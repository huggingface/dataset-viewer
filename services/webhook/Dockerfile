# build with
#   docker build -t some_tag_api -f Dockerfile ../..
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
COPY services/webhook/poetry.lock ./services/webhook/poetry.lock
COPY services/webhook/pyproject.toml ./services/webhook/pyproject.toml
COPY libs/libcommon ./libs/libcommon
COPY libs/libapi ./libs/libapi
WORKDIR /src/services/webhook/
RUN poetry install --no-cache
COPY services/webhook/src ./src
RUN poetry install --no-cache

ENTRYPOINT ["poetry", "run", "gunicorn", "'webhook.app:create_app()'", "-w $API_UVICORN_NUM_WORKERS", "-k uvicorn.workers.UvicornWorker", "-b $API_UVICORN_HOSTNAME:$API_UVICORN_PORT"]
