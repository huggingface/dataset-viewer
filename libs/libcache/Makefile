# environment variables for the commands (docker-compose, poetry)
export TEST_MONGO_PORT := 27020
export TEST_MONGO_CACHE_DATABASE := datasets_server_cache_test
export TEST_COMPOSE_PROJECT_NAME := libcache
# makefile variables
TEST_DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml

include ../../tools/Python.mk
include ../../tools/Docker.mk
