# environment variables for the commands (docker-compose, poetry)
export COMPOSE_PROJECT_NAME := first_rows
export MONGO_PORT := 27041
export CACHE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export QUEUE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml
#SAFETY_EXCEPTIONS :=

include ../../tools/Python.mk
include ../../tools/PythonTest.mk
include ../../tools/Docker.mk

.PHONY: run
run:
	poetry run python src/first_rows/main.py
