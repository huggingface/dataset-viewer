# build with
#   docker build -t some_tag_admin -f Dockerfile ../..
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
    && apt-get install -y build-essential make \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -U pip
RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /src
COPY services/admin/poetry.lock ./services/admin/poetry.lock
COPY services/admin/pyproject.toml ./services/admin/pyproject.toml
COPY libs/libcommon ./libs/libcommon
WORKDIR /src/services/admin/
RUN poetry install --no-cache
COPY services/admin/src ./src
RUN poetry install --no-cache

ENTRYPOINT ["poetry", "run", "python", "src/admin/main.py"]
