# environment variables for the commands (docker-compose, poetry)
export TEST_MONGO_PORT := 27021
export TEST_MONGO_QUEUE_DATABASE := datasets_server_queue_test
export TEST_COMPOSE_PROJECT_NAME := libqueue
# makefile variables
TEST_DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml

include ../../tools/Python.mk
include ../../tools/Docker.mk
