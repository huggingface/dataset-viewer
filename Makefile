# environment variables for the commands (docker-compose, poetry)
export LOCAL_CODE_MONGO_PORT := 27060
export LOCAL_CODE_SERVICE_ADMIN_PORT := 8081
export LOCAL_CODE_SERVICE_API_PORT := 8080
export LOCAL_CODE_SERVICE_REVERSE_PROXY_PORT := 8000
export LOCAL_CODE_COMPOSE_PROJECT_NAME := local-code

export REMOTE_IMAGES_MONGO_PORT := 27061
export REMOTE_IMAGES_SERVICE_ADMIN_PORT := 8181
export REMOTE_IMAGES_SERVICE_API_PORT := 8180
export REMOTE_IMAGES_SERVICE_REVERSE_PROXY_PORT := 8100
export REMOTE_IMAGES_COMPOSE_PROJECT_NAME := remote-images
# makefile variables
LOCAL_CODE_DOCKER_COMPOSE := ./tools/docker-compose-datasets-server-from-local-code.yml
REMOTE_IMAGES_DOCKER_COMPOSE := ./tools/docker-compose-datasets-server-from-remote-images.yml
DOCKER_IMAGES := ./infra/charts/datasets-server/docker-images.yaml

include tools/DockerRemoteImages.mk
include tools/Docker.mk

.PHONY: install
install:
	$(MAKE) -C e2e/ install
	$(MAKE) -C services/worker/ install
	$(MAKE) -C services/api/ install
	$(MAKE) -C services/admin/ install
	$(MAKE) -C libs/libcache/ install
	$(MAKE) -C libs/libqueue/ install
	$(MAKE) -C libs/libutils/ install

.PHONY: start-from-local-code
start-from-local-code:
	MONGO_PORT=${LOCAL_CODE_MONGO_PORT} SERVICE_ADMIN_PORT=${LOCAL_CODE_SERVICE_ADMIN_PORT} SERVICE_API_PORT=${LOCAL_CODE_SERVICE_API_PORT} SERVICE_REVERSE_PROXY_PORT=${LOCAL_CODE_SERVICE_REVERSE_PROXY_PORT} COMPOSE_PROJECT_NAME=${LOCAL_CODE_COMPOSE_PROJECT_NAME} DOCKER_COMPOSE=${LOCAL_CODE_DOCKER_COMPOSE} $(MAKE) down
	MONGO_PORT=${LOCAL_CODE_MONGO_PORT} SERVICE_ADMIN_PORT=${LOCAL_CODE_SERVICE_ADMIN_PORT} SERVICE_API_PORT=${LOCAL_CODE_SERVICE_API_PORT} SERVICE_REVERSE_PROXY_PORT=${LOCAL_CODE_SERVICE_REVERSE_PROXY_PORT} COMPOSE_PROJECT_NAME=${LOCAL_CODE_COMPOSE_PROJECT_NAME} DOCKER_COMPOSE=${LOCAL_CODE_DOCKER_COMPOSE} $(MAKE) up

.PHONY: stop-from-local-code
stop-from-local-code:
	MONGO_PORT=${LOCAL_CODE_MONGO_PORT} SERVICE_ADMIN_PORT=${LOCAL_CODE_SERVICE_ADMIN_PORT} SERVICE_API_PORT=${LOCAL_CODE_SERVICE_API_PORT} SERVICE_REVERSE_PROXY_PORT=${LOCAL_CODE_SERVICE_REVERSE_PROXY_PORT} COMPOSE_PROJECT_NAME=${LOCAL_CODE_COMPOSE_PROJECT_NAME} DOCKER_COMPOSE=${LOCAL_CODE_DOCKER_COMPOSE} $(MAKE) down

.PHONY: start-from-remote-images
start-from-remote-images:
	MONGO_PORT=${REMOTE_IMAGES_MONGO_PORT} SERVICE_ADMIN_PORT=${REMOTE_IMAGES_SERVICE_ADMIN_PORT} SERVICE_API_PORT=${REMOTE_IMAGES_SERVICE_API_PORT} SERVICE_REVERSE_PROXY_PORT=${REMOTE_IMAGES_SERVICE_REVERSE_PROXY_PORT} COMPOSE_PROJECT_NAME=${REMOTE_IMAGES_COMPOSE_PROJECT_NAME} DOCKER_COMPOSE=${REMOTE_IMAGES_DOCKER_COMPOSE} $(MAKE) down
	MONGO_PORT=${REMOTE_IMAGES_MONGO_PORT} SERVICE_ADMIN_PORT=${REMOTE_IMAGES_SERVICE_ADMIN_PORT} SERVICE_API_PORT=${REMOTE_IMAGES_SERVICE_API_PORT} SERVICE_REVERSE_PROXY_PORT=${REMOTE_IMAGES_SERVICE_REVERSE_PROXY_PORT} COMPOSE_PROJECT_NAME=${REMOTE_IMAGES_COMPOSE_PROJECT_NAME} DOCKER_COMPOSE=${REMOTE_IMAGES_DOCKER_COMPOSE} $(MAKE) up

.PHONY: stop-from-remote-images
stop-from-remote-images:
	MONGO_PORT=${REMOTE_IMAGES_MONGO_PORT} SERVICE_ADMIN_PORT=${REMOTE_IMAGES_SERVICE_ADMIN_PORT} SERVICE_API_PORT=${REMOTE_IMAGES_SERVICE_API_PORT} SERVICE_REVERSE_PROXY_PORT=${REMOTE_IMAGES_SERVICE_REVERSE_PROXY_PORT} COMPOSE_PROJECT_NAME=${REMOTE_IMAGES_COMPOSE_PROJECT_NAME} DOCKER_COMPOSE=${REMOTE_IMAGES_DOCKER_COMPOSE} $(MAKE) down

.PHONY: test
test:
	$(MAKE) -C services/admin/ test
	$(MAKE) -C services/worker/ test
	$(MAKE) -C services/api/ test
	$(MAKE) -C libs/libcache/ test
	$(MAKE) -C libs/libqueue/ test
	$(MAKE) -C libs/libutils/ test

.PHONY: coverage
coverage:
	$(MAKE) -C services/admin/ coverage
	$(MAKE) -C services/worker/ coverage
	$(MAKE) -C services/api/ coverage
	$(MAKE) -C libs/libcache/ coverage
	$(MAKE) -C libs/libqueue/ coverage
	$(MAKE) -C libs/libutils/ coverage

# Check that source code meets quality standards + security
.PHONY: quality
quality:
	$(MAKE) -C e2e/ quality
	$(MAKE) -C infra/charts/datasets-server/ quality
	$(MAKE) -C services/worker/ quality
	$(MAKE) -C services/api/ quality
	$(MAKE) -C services/admin/ quality
	$(MAKE) -C libs/libcache/ quality
	$(MAKE) -C libs/libqueue/ quality
	$(MAKE) -C libs/libutils/ quality

# Format source code automatically
.PHONY: style
style:
	$(MAKE) -C e2e/ style
	$(MAKE) -C services/worker/ style
	$(MAKE) -C services/api/ style
	$(MAKE) -C services/admin/ style
	$(MAKE) -C libs/libcache/ style
	$(MAKE) -C libs/libqueue/ style
	$(MAKE) -C libs/libutils/ style

.PHONY: e2e
e2e:
	$(MAKE) -C e2e/ e2e
