# environment variables for the commands (docker-compose, poetry)
export COMPOSE_PROJECT_NAME := first_rows
export MONGO_PORT := 27041
export CACHE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export CACHE_MONGO_DATABASE := datasets_server_cache_test
export COMMON_ASSETS_BASE_URL := http://localhost/assets
export COMMON_HF_ENDPOINT := https://hub-ci.huggingface.co
export COMMON_HF_TOKEN := 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
export FIRST_ROWS_MAX_NUMBER := 7
export QUEUE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export QUEUE_MONGO_DATABASE := datasets_server_queue_test
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml
#SAFETY_EXCEPTIONS :=

include ../../tools/Python.mk
include ../../tools/Docker.mk

.PHONY: run
run:
	poetry run python src/first_rows/main.py
