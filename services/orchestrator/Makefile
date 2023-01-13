# environment variables for the commands (docker compose, poetry)
export COMPOSE_PROJECT_NAME := orchestrator
export MONGO_PORT := 27050
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml

include ../../tools/Python.mk
include ../../tools/PythonAudit.mk
include ../../tools/PythonTest.mk
include ../../tools/Docker.mk

.PHONY: run-webapp
run-webapp:
	cd src/orchestrator; DAGSTER_HOME="$(shell pwd)/src/orchestrator" poetry run dagit --host 0.0.0.0 --port 3001 --path-prefix /admin/orchestrator

.PHONY: run-daemon
run-daemon:
	DAGSTER_HOME="$(shell pwd)/src/orchestrator" poetry run dagster-daemon run -f src/orchestrator/repo_assets.py
