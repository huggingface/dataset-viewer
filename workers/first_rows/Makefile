# environment variables for the commands (docker compose, poetry)
export COMPOSE_PROJECT_NAME := first_rows
export MONGO_PORT := 27041
export CACHE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export QUEUE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml

include ../../tools/Python.mk
include ../../tools/PythonTest.mk
include ../../tools/Docker.mk

.PHONY: run
run:
	poetry run python src/first_rows/main.py

.PHONY: pip-audit
pip-audit:
	bash -c "poetry run pip-audit -r <(poetry export -f requirements.txt --with dev | sed '/^pymongo==/,+110 d' | sed '/^requests==2.28.1 ;/,+2 d' | sed '/^kenlm @/d' | sed '/^trec-car-tools @/d' | sed '/^libcommon @/,+1 d')"
# ^ we remove problematic lines to have a working pip-audit. See https://github.com/pypa/pip-audit/issues/84#issuecomment-1326203111 for "requests"
