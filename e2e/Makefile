# environment variables for the commands (docker-compose, poetry)
export SERVICE_ADMIN_DOCKER_IMAGE := $(shell jq -r '.dockerImage.admin' ../infra/charts/datasets-server/docker-images.yaml)
export SERVICE_API_DOCKER_IMAGE := $(shell jq -r '.dockerImage.api' ../infra/charts/datasets-server/docker-images.yaml)
export SERVICE_DATASETS_WORKER_DOCKER_IMAGE := $(shell jq -r '.dockerImage.datasetsWorker' ../infra/charts/datasets-server/docker-images.yaml)
export SERVICE_REVERSE_PROXY_DOCKER_IMAGE := $(shell jq -r '.dockerImage.reverseProxy' ../infra/charts/datasets-server/docker-images.yaml)
export SERVICE_SPLITS_WORKER_DOCKER_IMAGE := $(shell jq -r '.dockerImage.splitsWorker' ../infra/charts/datasets-server/docker-images.yaml)
export SERVICE_REVERSE_PROXY_PORT := 8001
export TEST_MONGO_PORT := 27050
export TEST_MONGO_CACHE_DATABASE := datasets_server_cache_test
export TEST_MONGO_QUEUE_DATABASE := datasets_server_queue_test
export TEST_COMPOSE_PROJECT_NAME := e2e
# makefile variables
TEST_DOCKER_COMPOSE := ../tools/docker-compose-datasets-server-from-images.yml

include ../tools/Python.mk

.PHONY: e2e
e2e:
	make test
