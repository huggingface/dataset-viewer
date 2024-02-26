# environment variables for the commands (docker compose, poetry)
export COMPOSE_PROJECT_NAME := worker
export MONGO_PORT := 27040
export CACHE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export QUEUE_MONGO_URL := mongodb://localhost:${MONGO_PORT}

# Remove this config when all split-duckdb-index-010 entries have been computed
export DUCKDB_INDEX_CLI_PATH := /src/services/worker/duckdb

# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml

#export MONGO_PORT=27017
#export CACHE_MONGO_URL=mongodb://localhost:${MONGO_PORT}
#export QUEUE_MONGO_URL=mongodb://localhost:${MONGO_PORT}
##export DATABASE_MIGRATIONS_MONGO_URL=mongodb://localhost:${MONGO_PORT}
#export WORKER_JOB_TYPES_ONLY=dataset-config-names,config-parquet-and-info,config-info,config-split-names-from-info,config-split-names-from-streaming,split-descriptive-statistics
#export LOG_LEVEL=DEBUG
export WORKER_MAX_JOB_DURATION_SECONDS=1800
export WORKER_KILL_LONG_JOB_INTERVAL_SECONDS=1000
export DESCRIPTIVE_STATISTICS_MAX_PARQUET_SIZE_BYTES=5_000_000_000
export PARQUET_AND_INFO_MAX_DATASET_SIZE_BYTES=5_000_000_000
export API_HF_TIMEOUT_SECONDS=150
#export WORKER_UVICORN_PORT=8186
#
#export ASSETS_STORAGE_ROOT=/home/polina/.cache/storage/cached-assets
#export CACHED_ASSETS_STORAGE_ROOT=/home/polina/.cache/storage/cached-assets

include ../../tools/Python.mk
include ../../tools/PythonTest.mk
include ../../tools/Docker.mk

.PHONY: run
run:
	$(POETRY) run python src/worker/main.py
