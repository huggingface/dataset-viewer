# environment variables for the commands (docker-compose, poetry)
export COMPOSE_PROJECT_NAME := libqueue
export MONGO_PORT := 27021
export QUEUE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export QUEUE_MONGO_DATABASE := datasets_server_queue_test
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml

include ../../tools/Python.mk
include ../../tools/Docker.mk
