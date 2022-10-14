# environment variables for the commands (docker-compose, poetry)
export TEST_MONGO_PORT := 27041
export TEST_MONGO_CACHE_DATABASE := datasets_server_cache_test
export TEST_MONGO_QUEUE_DATABASE := datasets_server_queue_test
export TEST_ROWS_MAX_NUMBER := 5
export TEST_COMPOSE_PROJECT_NAME := worker_first_rows
export TEST_HF_ENDPOINT := https://hub-ci.huggingface.co
export TEST_HF_TOKEN := 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
# makefile variables
TEST_DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml
#SAFETY_EXCEPTIONS :=

# Ensure to specify HF_TOKEN when calling make test, ie HF_TOKEN=hf_app_xxx make test
include ../../tools/Python.mk
include ../../tools/Docker.mk

.PHONY: run
run:
	poetry run python src/worker_first_rows/main.py
