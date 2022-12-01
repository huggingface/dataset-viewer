export IMAGE_REVERSE_PROXY := $(shell jq -r '.dockerImage.reverseProxy' ${DOCKER_IMAGES})
export IMAGE_SERVICE_ADMIN := $(shell jq -r '.dockerImage.services.admin' ${DOCKER_IMAGES})
export IMAGE_SERVICE_API := $(shell jq -r '.dockerImage.services.api' ${DOCKER_IMAGES})
export IMAGE_WORKER_DATASETS_BASED := $(shell jq -r '.dockerImage.workers.datasets_based' ${DOCKER_IMAGES})
