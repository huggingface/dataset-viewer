# environment variables for the commands (docker-compose, poetry)
export COMPOSE_PROJECT_NAME := api
export MONGO_PORT := 27031
export CACHE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export CACHE_MONGO_DATABASE := datasets_server_cache_test
export QUEUE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export QUEUE_MONGO_DATABASE := datasets_server_queue_test
export API_UVICORN_HOST := localhost
export API_UVICORN_PORT := 8888
export COMMON_HF_ENDPOINT := http://${UVICORN_HOST}:${UVICORN_PORT}
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml

# Ensure to specify HF_TOKEN when calling make test, ie HF_TOKEN=hf_app_xxx make test
include ../../tools/Python.mk
include ../../tools/Docker.mk

.PHONY: run
run:
	poetry run python src/api/main.py

.PHONY: watch
watch:
	poetry run watchmedo auto-restart -d src/api -p "*.py" -R python src/api/main.py
