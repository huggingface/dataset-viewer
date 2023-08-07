# build with
#   docker build -t some_tag_storage-admin -f Dockerfile ../..
FROM ubuntu:jammy

# System deps:
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y rsync unzip wget curl glances \
    && rm -rf /var/lib/apt/lists/*
RUN curl https://rclone.org/install.sh | bash

ENTRYPOINT ["sleep", "infinity"]
