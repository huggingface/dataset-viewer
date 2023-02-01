# environment variables for the commands (docker compose, poetry)
export MONGO_PORT := 27060
export PORT_ADMIN := 8181
export PORT_API := 8180
export PORT_REVERSE_PROXY := 8100
export COMPOSE_PROJECT_NAME := datasets-server

# makefile variables
DOCKER_COMPOSE := ./tools/docker-compose-datasets-server.yml

include tools/Docker.mk

.PHONY: start
start:
	MONGO_PORT=${MONGO_PORT} ADMIN_UVICORN_PORT=${PORT_ADMIN} API_UVICORN_PORT=${PORT_API} PORT_REVERSE_PROXY=${PORT_REVERSE_PROXY} DOCKER_COMPOSE=${DOCKER_COMPOSE} $(MAKE) up

.PHONY: stop
stop:
	MONGO_PORT=${MONGO_PORT} ADMIN_UVICORN_PORT=${PORT_ADMIN} API_UVICORN_PORT=${PORT_API} PORT_REVERSE_PROXY=${PORT_REVERSE_PROXY} DOCKER_COMPOSE=${DOCKER_COMPOSE} $(MAKE) down

.PHONY: e2e
e2e:
	$(MAKE) -C e2e/ e2e

# for install, quality checks and tests of every job, lib, service or worker, see the Makefile in the corresponding folder
