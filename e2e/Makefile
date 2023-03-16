# environment variables for the commands (docker compose, poetry)
export ADMIN_UVICORN_NUM_WORKERS := 2
export ADMIN_UVICORN_PORT := 9081
export API_HF_JWT_ALGORITHM := EdDSA
export API_HF_JWT_PUBLIC_KEY_URL := https://hub-ci.huggingface.co/api/keys/jwt
export API_HF_TIMEOUT_SECONDS := 10
export API_UVICORN_NUM_WORKERS := 2
export API_UVICORN_PORT := 9080
export COMMON_HF_ENDPOINT := https://hub-ci.huggingface.co
export COMMON_HF_TOKEN := hf_app_datasets-server_token
export COMMON_LOG_LEVEL := DEBUG
export COMPOSE_PROJECT_NAME := e2e
export FIRST_ROWS_MAX_NUMBER := 4
export MONGO_PORT := 27050
export PARQUET_AND_DATASET_INFO_COMMITTER_HF_TOKEN := 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
export PORT_REVERSE_PROXY := 9000
export PROMETHEUS_MULTIPROC_DIR := /tmp
export QUEUE_SLEEP_TIME := 1
# ^ allow hub-ci to be slow to answer

# makefile variables
DOCKER_COMPOSE := ../tools/docker-compose-datasets-server.yml

include ../tools/Python.mk
include ../tools/PythonTest.mk
include ../tools/Docker.mk

.PHONY: e2e
e2e:
	make test

.PHONY: openapi
openapi:
	poetry run python -m openapi_spec_validator ../chart/static-files/openapi.json
