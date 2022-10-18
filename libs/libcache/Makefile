# environment variables for the commands (docker-compose, poetry)
export MONGO_PORT := 27020
export CACHE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export CACHE_MONGO_DATABASE := datasets_server_cache_test
export COMPOSE_PROJECT_NAME := libcache
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml

include ../../tools/Python.mk
include ../../tools/Docker.mk
