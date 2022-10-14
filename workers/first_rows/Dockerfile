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
    libicu-dev ffmpeg libavcodec-extra llvm \
    && rm -rf /var/lib/apt/lists/*
# Also install `libsndfile` in version `v1.0.30`. As the version in ubuntu stable for the moment is `v1.0.28`, we can build from scratch (see details here: https://github.com/libsndfile/libsndfile)
RUN apt-get update \
    && apt-get install -y autoconf autogen automake build-essential libasound2-dev libflac-dev libogg-dev libtool libvorbis-dev libopus-dev libmp3lame-dev libmpg123-dev pkg-config git;
WORKDIR /tmp
RUN git clone --depth=1 --branch=v1.0.30 https://github.com/libsndfile/libsndfile.git;
WORKDIR /tmp/libsndfile;
RUN /tmp/libsndfile/autogen.sh;
RUN /tmp/libsndfile/configure --enable-werror;
RUN make;
RUN make install;
RUN ldconfig;
WORKDIR /tmp
RUN rm -rf /tmp/libsndfile

RUN pip install -U --no-cache-dir pip
RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /src
COPY libs/libcache/dist ./libs/libcache/dist
COPY libs/libqueue/dist ./libs/libqueue/dist
COPY libs/libutils/dist ./libs/libutils/dist
COPY workers/first_rows/src ./workers/first_rows/src
COPY workers/first_rows/poetry.lock ./workers/first_rows/poetry.lock
COPY workers/first_rows/pyproject.toml ./workers/first_rows/pyproject.toml
COPY vendors ./vendors/
WORKDIR /src/workers/first_rows/
RUN poetry install

ENTRYPOINT ["poetry", "run", "python", "src/first_rows/main.py"]
