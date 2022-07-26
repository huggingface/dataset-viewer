# environment variables for the commands (docker-compose, poetry)
export SERVICE_ADMIN_PORT := 9081
export SERVICE_API_PORT := 9080
export SERVICE_REVERSE_PROXY_PORT := 9000
export TEST_MONGO_PORT := 27050
export TEST_MONGO_CACHE_DATABASE := datasets_server_cache_test
export TEST_MONGO_QUEUE_DATABASE := datasets_server_queue_test
export TEST_COMPOSE_PROJECT_NAME := e2e
# makefile variables
TEST_DOCKER_COMPOSE := ../tools/docker-compose-datasets-server-from-remote-images.yml
DOCKER_IMAGES := ../chart/docker-images.yaml

include ../tools/Python.mk
include ../tools/DockerRemoteImages.mk
include ../tools/Docker.mk

.PHONY: e2e
e2e:
	make test
