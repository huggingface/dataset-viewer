# environment variables for the commands (docker compose, poetry)
export MONGO_PORT := 27020
export CACHE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export QUEUE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export METRICS_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export COMPOSE_PROJECT_NAME := libcommon
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml

include ../../tools/Python.mk
include ../../tools/PythonTest.mk
include ../../tools/Docker.mk
