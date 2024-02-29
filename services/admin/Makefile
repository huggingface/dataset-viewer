# environment variables for the commands (docker compose, poetry)
export COMPOSE_PROJECT_NAME := admin
export MONGO_PORT := 27030
export CACHE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export QUEUE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml
TEST_PATH ?= tests

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
