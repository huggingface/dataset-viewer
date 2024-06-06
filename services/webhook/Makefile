# environment variables for the commands (docker compose, poetry)
export COMPOSE_PROJECT_NAME := webhook
export MONGO_PORT := 27036
export CACHE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export QUEUE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml
TEST_PATH ?= tests

include ../../tools/Python.mk
include ../../tools/Docker.mk

.PHONY: run
run:
	$(POETRY) run python src/webhook/main.py

.PHONY: watch
watch:
	$(POETRY) run watchmedo auto-restart -d src/webhook -p "*.py" -R python src/webhook/main.py

# override the default test target to test prometheus depending on the environment
# we cannot set the env var with pytest.MonkeyPatch, it's too late
.PHONY: test
test:
	$(MAKE) down
	$(MAKE) up
	$(POETRY) run python -m pytest --memray -vv -x ${ADDOPTS} $(TEST_PATH)
	rm -rf /tmp/webhook.prometheus
	mkdir /tmp/webhook.prometheus
	PROMETHEUS_MULTIPROC_DIR=/tmp/webhook.prometheus $(POETRY) run python -m pytest --memray -vv -x -k "test_metrics" ${ADDOPTS} $(TEST_PATH)
	rm -rf /tmp/webhook.prometheus
	$(MAKE) down
