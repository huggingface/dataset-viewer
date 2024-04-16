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
export ROWS_UVICORN_NUM_WORKERS := 2
export ROWS_UVICORN_PORT := 9082
export SEARCH_UVICORN_NUM_WORKERS := 2
export SEARCH_UVICORN_PORT := 9083
export COMMON_HF_ENDPOINT := https://hub-ci.huggingface.co
export COMMON_HF_TOKEN := hf_app_datasets-server_token
# ^ hard coded, see e2e/tests/fixtures/hub.py
export COMMON_BLOCKED_DATASETS := DVUser/blocked-*
export LOG_LEVEL := DEBUG
export MONGO_PORT := 27050
export PARQUET_AND_INFO_COMMITTER_HF_TOKEN := hf_app_datasets-server-parquet-converter_token
export DUCKDB_INDEX_COMMITTER_HF_TOKEN := hf_app_datasets-server-parquet-converter_token
export PORT_REVERSE_PROXY := 9000
export PROMETHEUS_MULTIPROC_DIR := /tmp
export WORKER_SLEEP_SECONDS := 1
export WORKER_UVICORN_NUM_WORKERS := 2
export WORKER_UVICORN_PORT := 9086

# see https://github.com/huggingface/moon-landing/blob/main/server/scripts/staging-seed-db.ts
export ADMIN_HF_ORGANIZATION := valid_org
export E2E_ADMIN_USER_TOKEN := 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

export COMPOSE_PROJECT_NAME := e2e

# set the secret environment variables by your own (ie. in a .env file, and source it):
export ASSETS_BASE_URL := https://datasets-server-test.us.dev.moon.huggingface.tech/assets
export ASSETS_STORAGE_PROTOCOL := s3
export ASSETS_STORAGE_ROOT := hf-datasets-server-statics-test/assets
export CACHED_ASSETS_BASE_URL := https://datasets-server-test.us.dev.moon.huggingface.tech/cached-assets
export CACHED_ASSETS_STORAGE_PROTOCOL := s3
export CACHED_ASSETS_STORAGE_ROOT := hf-datasets-server-statics-test/cached-assets
export CLOUDFRONT_KEY_PAIR_ID := K3814DK2QUJ71H
# CLOUDFRONT_PRIVATE_KEY <- secret
export S3_REGION_NAME := us-east-1
# S3_ACCESS_KEY_ID       <- secret
# S3_SECRET_ACCESS_KEY   <- secret

# makefile variables
DOCKER_COMPOSE := ../tools/docker-compose-dataset-viewer.yml

include ../tools/Python.mk
include ../tools/PythonTest.mk
include ../tools/Docker.mk

.PHONY: e2e
e2e:
	make test
