# environment variables for the commands (docker compose, poetry)
export COMPOSE_PROJECT_NAME := admin
export MONGO_PORT := 27030
export CACHE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export QUEUE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export METRICS_MONGO_URL := mongodb://localhost:${MONGO_PORT}
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml
TEST_PATH ?= tests

include ../../tools/Python.mk
include ../../tools/Docker.mk

.PHONY: run
run:
	poetry run python src/admin/main.py

.PHONY: watch
watch:
	poetry run watchmedo auto-restart -d src/admin -p "*.py" -R python src/admin/main.py

# override the default test target to test prometheus depending on the environment
# we cannot set the env var with pytest.MonkeyPatch, it's too late
.PHONY: test
test:
	$(MAKE) up
	poetry run python -m pytest -vv -x ${ADDOPTS} $(TEST_PATH)
	rm -rf /tmp/admin.prometheus
	mkdir /tmp/admin.prometheus
	PROMETHEUS_MULTIPROC_DIR=/tmp/admin.prometheus poetry run python -m pytest -vv -x -k "test_metrics or test_prometheus" ${ADDOPTS} $(TEST_PATH)
	rm -rf /tmp/admin.prometheus
	$(MAKE) down

.PHONY: coverage
coverage:
	$(MAKE) up
	poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term $(TEST_PATH)
	$(MAKE) down
