# environment variables for the commands (docker-compose, poetry)
export SERVICE_ADMIN_PORT := 9081
export SERVICE_API_PORT := 9080
export SERVICE_REVERSE_PROXY_PORT := 9000
export TEST_MONGO_PORT := 27050
export TEST_ROWS_MAX_NUMBER := 4
export TEST_COMPOSE_PROJECT_NAME := e2e
export TEST_EXTERNAL_AUTH_URL := https://huggingface.co/api/datasets/%s/auth-check
# makefile variables
TEST_DOCKER_COMPOSE := ../tools/docker-compose-datasets-server-from-remote-images.yml
DOCKER_IMAGES := ../chart/docker-images.yaml

include ../tools/Python.mk
include ../tools/DockerRemoteImages.mk
include ../tools/Docker.mk

.PHONY: e2e
e2e:
	make test
