# environment variables for the commands (docker compose, poetry)
export ADMIN_UVICORN_NUM_WORKERS := 2
export ADMIN_UVICORN_PORT := 9081
export API_HF_JWT_ALGORITHM := EdDSA
export API_HF_JWT_PUBLIC_KEY_URL := https://hub-ci.huggingface.co/api/keys/jwt
# ^ it's not tested in the e2e tests, but it's good to ensure we're able to fetch it at least.
export ADMIN_HF_TIMEOUT_SECONDS := 10
export API_HF_TIMEOUT_SECONDS := 10
# ^ allow hub-ci to be slow to answer
export API_UVICORN_NUM_WORKERS := 2
export API_UVICORN_PORT := 9080
export COMMON_HF_ENDPOINT := https://hub-ci.huggingface.co
export COMMON_HF_TOKEN := hf_app_datasets-server_token
# ^ hard coded, see e2e/tests/fixtures/hub.py
export LOG_LEVEL := DEBUG
export FIRST_ROWS_MAX_NUMBER := 4
export MONGO_PORT := 27050
export PARQUET_AND_INFO_COMMITTER_HF_TOKEN := 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
export PORT_REVERSE_PROXY := 9000
export PROMETHEUS_MULTIPROC_DIR := /tmp
export WORKER_SLEEP_SECONDS := 1

export COMPOSE_PROJECT_NAME := e2e

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
