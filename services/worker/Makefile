# environment variables for the commands (docker-compose, poetry)
export TEST_MONGO_PORT := 27032
export TEST_MONGO_CACHE_DATABASE := datasets_server_cache_test
export TEST_MONGO_QUEUE_DATABASE := datasets_server_queue_test
export TEST_ROWS_MAX_NUMBER := 5
export TEST_COMPOSE_PROJECT_NAME := worker
# makefile variables
TEST_DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml
# pillow <9.0.0
PILLOW_EXCEPTIONS := -i 44525 -i 44524 -i 44486 -i 44485 -i 45356 -i 44487
SAFETY_EXCEPTIONS := $(PILLOW_EXCEPTIONS)

# Ensure to specify HF_TOKEN when calling make test, ie HF_TOKEN=hf_app_xxx make test
include ../../tools/Python.mk

.PHONY: run
run:
	poetry run python src/worker/main.py

.PHONY: datasets-worker
datasets-worker:
	WORKER_QUEUE=datasets poetry run python src/worker/main.py

.PHONY: splits-worker
splits-worker:
	WORKER_QUEUE=splits poetry run python src/worker/main.py
