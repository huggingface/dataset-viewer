# environment variables for the commands (docker-compose, poetry)
export COMPOSE_PROJECT_NAME := parquet
export MONGO_PORT := 27042
export CACHE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export QUEUE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml
#SAFETY_EXCEPTIONS :=

include ../../tools/Python.mk
include ../../tools/PythonTest.mk
include ../../tools/Docker.mk

.PHONY: run
run:
	poetry run python src/parquet/main.py

.PHONY: pip-audit
pip-audit:
	bash -c "poetry run pip-audit -r <(poetry export -f requirements.txt --with dev | sed '/^pymongo==/,+110 d' | sed '/^requests==2.28.1 ;/,+2 d' | sed '/^kenlm @/d' | sed '/^trec-car-tools @/d')"
# ^ we remove problematic lines to have a working pip-audit. See https://github.com/pypa/pip-audit/issues/84#issuecomment-1326203111 for "requests"
