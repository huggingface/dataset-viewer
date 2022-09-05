# environment variables for the commands (docker-compose, poetry)
export TEST_MONGO_PORT := 27030
export TEST_MONGO_CACHE_DATABASE := datasets_server_cache_test
export TEST_MONGO_QUEUE_DATABASE := datasets_server_queue_test
export TEST_COMPOSE_PROJECT_NAME := admin
export TEST_HF_ENDPOINT := https://hub-ci.huggingface.co
# makefile variables
TEST_DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml

# Ensure to specify HF_TOKEN when calling make test, ie HF_TOKEN=hf_app_xxx make test
include ../../tools/Python.mk
include ../../tools/Docker.mk
include ./Scripts.mk

.PHONY: run
run:
	poetry run python src/admin/main.py

.PHONY: watch
watch:
	poetry run watchmedo auto-restart -d src/admin -p "*.py" -R python src/admin/main.py
