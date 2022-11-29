# environment variables for the commands (docker compose, poetry)
export MONGO_PORT := 27060
export PORT_ADMIN := 8181
export PORT_API := 8180
export PORT_REVERSE_PROXY := 8100
export COMPOSE_PROJECT_NAME := datasets-server

# makefile variables
DOCKER_COMPOSE := ./tools/docker-compose-datasets-server.yml
DOCKER_IMAGES := ./chart/docker-images.yaml

include tools/DockerRemoteImages.mk
include tools/Docker.mk

.PHONY: install
install:
	$(MAKE) -C e2e/ install
	$(MAKE) -C services/api/ install
	$(MAKE) -C services/admin/ install
	$(MAKE) -C libs/libcommon/ install
	$(MAKE) -C workers/first_rows install
	$(MAKE) -C workers/splits install

.PHONY: start
start:
	MONGO_PORT=${MONGO_PORT} ADMIN_UVICORN_PORT=${PORT_ADMIN} API_UVICORN_PORT=${PORT_API} PORT_REVERSE_PROXY=${PORT_REVERSE_PROXY} DOCKER_COMPOSE=${DOCKER_COMPOSE} $(MAKE) up

.PHONY: stop
stop:
	MONGO_PORT=${MONGO_PORT} ADMIN_UVICORN_PORT=${PORT_ADMIN} API_UVICORN_PORT=${PORT_API} PORT_REVERSE_PROXY=${PORT_REVERSE_PROXY} DOCKER_COMPOSE=${DOCKER_COMPOSE} $(MAKE) down

.PHONY: test
test:
	$(MAKE) -C services/admin/ test
	$(MAKE) -C services/api/ test
	$(MAKE) -C libs/libcommon/ test
	$(MAKE) -C workers/first_rows test
	$(MAKE) -C workers/splits test

.PHONY: coverage
coverage:
	$(MAKE) -C services/admin/ coverage
	$(MAKE) -C services/api/ coverage
	$(MAKE) -C libs/libcommon/ coverage
	$(MAKE) -C workers/first_rows coverage
	$(MAKE) -C workers/splits coverage

# Check that source code meets quality standards + security
.PHONY: quality
quality:
	$(MAKE) -C e2e/ quality
	$(MAKE) -C e2e/ openapi
	$(MAKE) -C chart/ quality
	$(MAKE) -C services/api/ quality
	$(MAKE) -C services/admin/ quality
	$(MAKE) -C libs/libcommon/ quality
	$(MAKE) -C workers/first_rows quality
	$(MAKE) -C workers/splits quality

# Format source code automatically
.PHONY: style
style:
	$(MAKE) -C e2e/ style
	$(MAKE) -C services/api/ style
	$(MAKE) -C services/admin/ style
	$(MAKE) -C libs/libcommon/ style
	$(MAKE) -C workers/first_rows style
	$(MAKE) -C workers/splits style

.PHONY: e2e
e2e:
	$(MAKE) -C e2e/ e2e
