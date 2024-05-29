# environment variables for the commands (docker compose, poetry)
export COMPOSE_PROJECT_NAME := search
export MONGO_PORT := 27033
export CACHE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export QUEUE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export HF_HUB_ENABLE_HF_TRANSFER := 1
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml
TEST_PATH ?= tests

include ../../tools/Python.mk
include ../../tools/Docker.mk

.PHONY: run
run:
	$(POETRY) run python src/search/main.py

.PHONY: watch
watch:
	$(POETRY) run watchmedo auto-restart -d src/search -p "*.py" -R python src/search/main.py

# override the default test target to test prometheus depending on the environment
# we cannot set the env var with pytest.MonkeyPatch, it's too late
.PHONY: test
test:
	$(MAKE) down
	$(MAKE) up
	$(POETRY) run python -m pytest --memray -vv -x ${ADDOPTS} $(TEST_PATH)
	rm -rf /tmp/search.prometheus
	mkdir /tmp/search.prometheus
	PROMETHEUS_MULTIPROC_DIR=/tmp/search.prometheus $(POETRY) run python -m pytest --memray -vv -x -k "test_metrics" ${ADDOPTS} $(TEST_PATH)
	rm -rf /tmp/search.prometheus
	$(MAKE) down
