# environment variables for the commands (docker compose, poetry)
export COMPOSE_PROJECT_NAME := admin
export MONGO_PORT := 27030
export CACHE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export QUEUE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml
TEST_PATH ?= tests

#export MONGO_PORT=27017
#export CACHE_MONGO_URL=mongodb://localhost:${MONGO_PORT}
#export QUEUE_MONGO_URL=mongodb://localhost:${MONGO_PORT}
#export DATABASE_MIGRATIONS_MONGO_URL=mongodb://localhost:${MONGO_PORT}
#export WORKER_STATE_FILE_PATH=worker.json
#export LOG_LEVEL=DEBUG
#export PARQUET_AND_INFO_COMMITTER_HF_TOKEN="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"
#export ADMIN_UVICORN_PORT=8100
#export ASSETS_STORAGE_ROOT=/home/polina/.cache/storage

include ../../tools/Python.mk
include ../../tools/Docker.mk

.PHONY: run
run:
	$(POETRY) run python src/admin/main.py

.PHONY: watch
watch:
	$(POETRY) run watchmedo auto-restart -d src/admin -p "*.py" -R python src/admin/main.py

# override the default test target to test prometheus depending on the environment
# we cannot set the env var with pytest.MonkeyPatch, it's too late
.PHONY: test
test:
	$(MAKE) down
	$(MAKE) up
	$(POETRY) run python -m pytest -vv -x ${ADDOPTS} $(TEST_PATH)
	rm -rf /tmp/admin.prometheus
	mkdir /tmp/admin.prometheus
	PROMETHEUS_MULTIPROC_DIR=/tmp/admin.prometheus $(POETRY) run python -m pytest -vv -x -k "test_metrics" ${ADDOPTS} $(TEST_PATH)
	rm -rf /tmp/admin.prometheus
	$(MAKE) down
