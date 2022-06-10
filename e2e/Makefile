include ../tools/Common.mk

export SERVICE_ADMIN_DOCKER_IMAGE := $(shell jq -r '.dockerImage.admin' ../infra/charts/datasets-server/docker-images.yaml)
export SERVICE_API_DOCKER_IMAGE := $(shell jq -r '.dockerImage.api' ../infra/charts/datasets-server/docker-images.yaml)
export SERVICE_DATASETS_WORKER_DOCKER_IMAGE := $(shell jq -r '.dockerImage.datasetsWorker' ../infra/charts/datasets-server/docker-images.yaml)
export SERVICE_REVERSE_PROXY_DOCKER_IMAGE := $(shell jq -r '.dockerImage.reverseProxy' ../infra/charts/datasets-server/docker-images.yaml)
export SERVICE_SPLITS_WORKER_DOCKER_IMAGE := $(shell jq -r '.dockerImage.splitsWorker' ../infra/charts/datasets-server/docker-images.yaml)

.PHONY: e2e
e2e:
	docker-compose -f ./docker-compose.yml down
	docker volume rm e2e_assets e2e_mongo e2e_reverse-proxy-cache
	docker-compose -f ./docker-compose.yml up -d
	poetry run python -m pytest -x tests
	docker-compose -f ./docker-compose.yml down
