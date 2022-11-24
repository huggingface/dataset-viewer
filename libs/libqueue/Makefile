# environment variables for the commands (docker-compose, poetry)
export COMPOSE_PROJECT_NAME := libqueue
export MONGO_PORT := 27021
export QUEUE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml

include ../../tools/Python.mk
include ../../tools/PythonTest.mk
include ../../tools/Docker.mk

.PHONY: pip-audit
pip-audit:
	bash -c "poetry run pip-audit -r <(poetry export -f requirements.txt --with dev | sed '/^pymongo==4.3.3 ;/,+74 d')"
# ^ we remove problematic lines to have a working pip-audit. See https://github.com/pypa/pip-audit/issues/84#issuecomment-1326203111 for "requests"
