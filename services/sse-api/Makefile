# environment variables for the commands (docker compose, poetry)
export COMPOSE_PROJECT_NAME := sse-api
export MONGO_PORT := 27035
export CACHE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml
TEST_PATH ?= tests

include ../../tools/Python.mk
include ../../tools/PythonTest.mk
include ../../tools/Docker.mk

.PHONY: run
run:
	poetry run python src/sse_api/main.py

.PHONY: watch
watch:
	poetry run watchmedo auto-restart -d src/sse_api -p "*.py" -R python src/sse_api/main.py
