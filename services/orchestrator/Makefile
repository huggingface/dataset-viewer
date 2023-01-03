# environment variables for the commands (docker compose, poetry)
export COMPOSE_PROJECT_NAME := orchestrator
export MONGO_PORT := 27050
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml

include ../../tools/Python.mk
include ../../tools/PythonAudit.mk
include ../../tools/PythonTest.mk
include ../../tools/Docker.mk

.PHONY: run
run:
	poetry run python src/orchestrator/main.py
