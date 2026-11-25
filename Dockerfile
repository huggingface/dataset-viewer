# Multi-stage Dockerfile for all dataset-viewer services and jobs
# Build with: docker build --target <service_name> -t <tag> .

ARG PYTHON_VERSION=3.14.5
FROM python:${PYTHON_VERSION}-slim AS viewer

# Install Rust and minimal build deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain and maturin
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && . $HOME/.cargo/env \
    && pip install maturin \
    && rustc --version \
    && cargo --version
# Add cargo bin dir to PATH (so maturin + cargo available globally)
ENV PATH="/root/.cargo/bin:${PATH}"

# Build libviewer
COPY libs/libviewer /src/libs/libviewer
WORKDIR /src/libs/libviewer
RUN maturin build --release --strip --out /tmp/dist

# FFmpeg libraries for torchcodec, built with only the decoders, demuxers and filters the viewer needs.
# User-uploaded media reach the FFmpeg decoders (avformat_find_stream_info decodes frames before any codec check),
# so every compiled decoder is attack surface: Debian's build ships hundreds of exotic ones (TAK, VMnc, Cook...)
# whose security fixes Debian postpones until they land in the upstream 7.1 branch.
# Stay on FFmpeg 7 (libavutil 59): torchcodec 0.15 supports FFmpeg 4 to 8 only.
FROM python:${PYTHON_VERSION}-slim AS ffmpeg
# head of the upstream release/7.1 branch on 2026-11-17
ARG FFMPEG_COMMIT=2f5e4b9521db2ae309e5c2dd6e409fab27e1ad99
ARG FFMPEG_SHA256=82ce458e07914714c28028caed59326d1ac1b97fdf800201934162cc9434ff69
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates build-essential nasm pkg-config patch \
      libdav1d-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src/ffmpeg
RUN curl -fsSL -o /tmp/ffmpeg.tar.gz "https://github.com/FFmpeg/FFmpeg/archive/${FFMPEG_COMMIT}.tar.gz" \
    && echo "${FFMPEG_SHA256}  /tmp/ffmpeg.tar.gz" | sha256sum -c - \
    && tar xzf /tmp/ffmpeg.tar.gz --strip-components=1 \
    && rm /tmp/ffmpeg.tar.gz
# upstream master fixes for code that this build keeps, not yet in release/7.1
COPY tools/ffmpeg/patches /src/ffmpeg-patches
RUN for p in /src/ffmpeg-patches/*.patch; do patch -p1 < "$p"; done
RUN ./configure \
      --prefix=/opt/ffmpeg \
      --enable-shared --disable-static \
      --disable-programs --disable-doc --disable-network --disable-autodetect \
      --disable-everything \
      --enable-libdav1d \
      --enable-protocol=file,pipe \
      --enable-demuxer=aac,ac3,aiff,amr,asf,au,avi,caf,eac3,flac,matroska,mov,mp3,mpegps,mpegts,ogg,w64,wav \
      --enable-parser=aac,aac_latm,ac3,av1,flac,h264,hevc,mjpeg,mpeg4video,mpegaudio,mpegvideo,opus,vorbis,vp8,vp9 \
      --enable-decoder=aac,aac_latm,ac3,adpcm_ima_wav,adpcm_ms,alac,amrnb,amrwb,eac3,flac,mp2,mp2float,mp3,mp3float \
      --enable-decoder=opus,vorbis,wmav1,wmav2,'pcm_*' \
      --enable-decoder=h264,hevc,libdav1d,mjpeg,mpeg1video,mpeg2video,mpeg4,vp8,vp9 \
      --enable-encoder=pcm_s16le,pcm_s32le,pcm_f32le \
      --enable-muxer=wav \
      --enable-filter=abuffer,abuffersink,aformat,anull,aresample,buffer,buffersink,crop,format,hflip,null,rotate \
      --enable-filter=scale,transpose,vflip \
    && make -j"$(nproc)" \
    && make install

# Base stage with shared setup
FROM python:${PYTHON_VERSION}-slim AS libcommon

# System dependencies.
# APT_SNAPSHOT is bumped weekly (see cd.yml) to bust this layer's build cache so
# Debian security updates reach the published images.
ARG APT_SNAPSHOT=base
RUN echo "apt snapshot: ${APT_SNAPSHOT}" \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y unzip wget procps htop libdav1d7 libsndfile1 libjpeg-dev zlib1g-dev \
      libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libxrender1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ffmpeg /opt/ffmpeg/lib/*.so* /usr/local/lib/
RUN ldconfig

# Common environment variables
ARG POETRY_VERSION=2.1.4
ENV USER=app
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PATH="$PATH:/root/.local/bin"

# Install pip and poetry
RUN pip install -U pip && pip install "poetry==${POETRY_VERSION}"

# Add dummy pyproject.toml for libviewer so that poetry install
# can resolve it but we later overwrite libviewer with the prebuilt
# wheel.
RUN mkdir -p /src/libs/libviewer && \
    printf "[project]\nname = \"libviewer\"\nversion = \"0.0.0\"\n" > /src/libs/libviewer/pyproject.toml

# Install libcommon's dependencies but not libcommon itself.
COPY libs/libcommon/poetry.lock \
     libs/libcommon/pyproject.toml \
     /src/libs/libcommon/
WORKDIR /src/libs/libcommon
RUN poetry install --no-cache --no-root

# Add libcommon source but do not install
COPY libs/libcommon /src/libs/libcommon
RUN poetry install --no-cache

# Base image for all services
FROM libcommon AS libapi
COPY libs/libapi /src/libs/libapi
WORKDIR /src/libs/libapi
RUN poetry install --no-cache

# Below are the actual API services which depend on libapi and libcommon.
# Since the majority of the dependencies are already installed in the
# `common` stage we let poetry to handle the rest.

# API service
FROM libapi AS api
COPY services/api /src/services/api
WORKDIR /src/services/api
RUN poetry install --no-cache
CMD ["poetry", "run", "python", "src/api/main.py"]

# Admin service
FROM libapi AS admin
COPY services/admin /src/services/admin
WORKDIR /src/services/admin
RUN poetry install --no-cache
CMD ["poetry", "run", "python", "src/admin/main.py"]

# Rows service
FROM libapi AS rows
COPY services/rows /src/services/rows
WORKDIR /src/services/rows
RUN poetry install --no-cache
# Install libviewer wheel built in the viewer stage
COPY --from=viewer /tmp/dist /tmp/dist
RUN pip install /tmp/dist/libviewer-*.whl
CMD ["poetry", "run", "python", "src/rows/main.py"]

# Search service
FROM libapi AS search
ARG DUCKDB_INDEX_EXTENSIONS_DIRECTORY="/tmp/duckdb-extensions"
COPY services/search /src/services/search
WORKDIR /src/services/search
RUN poetry install --no-cache
# Install libviewer wheel built in the viewer stage
COPY --from=viewer /tmp/dist /tmp/dist
RUN pip install /tmp/dist/libviewer-*.whl
RUN python -c "import duckdb; duckdb.execute(\"SET extension_directory = '$DUCKDB_INDEX_EXTENSIONS_DIRECTORY'; INSTALL 'fts';\")"
CMD ["poetry", "run", "python", "src/search/main.py"]

# SSE API service
FROM libapi AS sse-api
COPY services/sse-api /src/services/sse-api
WORKDIR /src/services/sse-api
RUN poetry install --no-cache
CMD ["poetry", "run", "python", "src/sse_api/main.py"]

# Webhook service
FROM libapi AS webhook
COPY services/webhook /src/services/webhook
WORKDIR /src/services/webhook
RUN poetry install --no-cache
CMD ["poetry", "run", "python", "src/webhook/main.py"]

# Worker service
FROM libcommon AS worker
COPY services/worker /src/services/worker
WORKDIR /src/services/worker
# presidio-analyzer > spacy > thinc doesn't ship aarch64 wheels so need to compile
RUN if [ "$(uname -m)" = "aarch64" ]; then \
      apt-get update && apt-get install -y build-essential && \
      rm -rf /var/lib/apt/lists/*; \
    fi
RUN poetry install --no-cache
# Install libviewer wheel built in the viewer stage
COPY --from=viewer /tmp/dist /tmp/dist
RUN pip install /tmp/dist/libviewer-*.whl
RUN python -m spacy download en_core_web_lg
CMD ["poetry", "run", "python", "src/worker/main.py"]

# Cache maintenance job
FROM libcommon AS cache_maintenance
COPY jobs/cache_maintenance /src/jobs/cache_maintenance
WORKDIR /src/jobs/cache_maintenance
RUN poetry install --no-cache
CMD ["poetry", "run", "python", "src/cache_maintenance/main.py"]

# MongoDB migration job
FROM libcommon AS mongodb_migration
COPY jobs/mongodb_migration /src/jobs/mongodb_migration
WORKDIR /src/jobs/mongodb_migration
RUN poetry install --no-cache
# Install libviewer wheel built in the viewer stage
COPY --from=viewer /tmp/dist /tmp/dist
RUN pip install /tmp/dist/libviewer-*.whl
CMD ["poetry", "run", "python", "src/mongodb_migration/main.py"]
