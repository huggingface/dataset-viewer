# environment variables for the commands (docker compose, poetry)
export COMPOSE_PROJECT_NAME := worker
export MONGO_PORT := 27040
export CACHE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export QUEUE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export DUCKDB_INDEX_CLI_PATH := /src/services/worker/duckdb

# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml

include ../../tools/Python.mk
include ../../tools/PythonTest.mk
include ../../tools/Docker.mk

.PHONY: run
run:
	$(POETRY) run python src/worker/main.py
