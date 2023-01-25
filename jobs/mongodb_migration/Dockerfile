# build with
#   docker build -t some_tag_admin -f Dockerfile ../..
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

# System deps:
RUN apt-get update \
    && apt-get install -y build-essential python3-dev make \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -U --no-cache-dir pip
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /src
COPY jobs/mongodb_migration/poetry.lock ./jobs/mongodb_migration/poetry.lock
COPY jobs/mongodb_migration/pyproject.toml ./jobs/mongodb_migration/pyproject.toml
COPY libs/libcommon ./libs/libcommon
WORKDIR /src/jobs/mongodb_migration/
RUN poetry install --no-cache
COPY jobs/mongodb_migration/src ./src
RUN poetry install --no-cache

ENTRYPOINT ["poetry", "run", "python", "src/mongodb_migration/main.py"]
