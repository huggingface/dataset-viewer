# environment variables for the commands (docker compose, poetry)
export MONGO_PORT := 27060
export PORT_ADMIN := 8181
export PORT_API := 8180
export PORT_REVERSE_PROXY := 8100

# environment variables per target
start: export COMPOSE_PROJECT_NAME := datasets-server
stop: export COMPOSE_PROJECT_NAME := datasets-server
dev-start: export COMPOSE_PROJECT_NAME := dev-datasets-server
dev-stop: export COMPOSE_PROJECT_NAME := dev-datasets-server

# makefile variables per target
start: DOCKER_COMPOSE := ./tools/docker-compose-datasets-server.yml
stop: DOCKER_COMPOSE := ./tools/docker-compose-datasets-server.yml
dev-start: DOCKER_COMPOSE := ./tools/docker-compose-dev-datasets-server.yml
dev-stop: DOCKER_COMPOSE := ./tools/docker-compose-dev-datasets-server.yml

include tools/Docker.mk

.PHONY: start
start:
	MONGO_PORT=${MONGO_PORT} ADMIN_UVICORN_PORT=${PORT_ADMIN} API_UVICORN_PORT=${PORT_API} PORT_REVERSE_PROXY=${PORT_REVERSE_PROXY} DOCKER_COMPOSE=${DOCKER_COMPOSE} $(MAKE) up

.PHONY: stop
stop:
	MONGO_PORT=${MONGO_PORT} ADMIN_UVICORN_PORT=${PORT_ADMIN} API_UVICORN_PORT=${PORT_API} PORT_REVERSE_PROXY=${PORT_REVERSE_PROXY} DOCKER_COMPOSE=${DOCKER_COMPOSE} $(MAKE) down

.PHONY: dev-start
dev-start:
	MONGO_PORT=${MONGO_PORT} ADMIN_UVICORN_PORT=${PORT_ADMIN} API_UVICORN_PORT=${PORT_API} PORT_REVERSE_PROXY=${PORT_REVERSE_PROXY} DOCKER_COMPOSE=${DOCKER_COMPOSE} $(MAKE) up

.PHONY: dev-stop
dev-stop:
	MONGO_PORT=${MONGO_PORT} ADMIN_UVICORN_PORT=${PORT_ADMIN} API_UVICORN_PORT=${PORT_API} PORT_REVERSE_PROXY=${PORT_REVERSE_PROXY} DOCKER_COMPOSE=${DOCKER_COMPOSE} $(MAKE) down

.PHONY: e2e
e2e:
	$(MAKE) -C e2e/ e2e

# for install, quality checks and tests of every job, lib, service or worker, see the Makefile in the corresponding folder
