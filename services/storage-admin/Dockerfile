# build with
#   docker build -t some_tag_storage-admin -f Dockerfile ../..
FROM ubuntu:jammy

# System deps:
RUN apt-get update \
    && apt-get install -y rsync unzip wget curl \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["sleep", "infinity"]
