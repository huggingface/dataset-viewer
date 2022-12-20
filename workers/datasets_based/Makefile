# environment variables for the commands (docker compose, poetry)
export COMPOSE_PROJECT_NAME := datasets_based
export MONGO_PORT := 27040
export CACHE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export QUEUE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml

include ../../tools/Python.mk
include ../../tools/PythonTest.mk
include ../../tools/Docker.mk

.PHONY: run
run:
	poetry run python src/datasets_based/main.py

.PHONY: pip-audit
pip-audit:
	bash -c "poetry run pip-audit --ignore-vuln GHSA-47fc-vmwq-366v --ignore-vuln GHSA-hcpj-qp55-gfph -r <(poetry export -f requirements.txt --with dev | sed '/^pymongo==/,+109 d' | sed '/^requests==2.28.1 ;/,+2 d' | sed '/^kenlm @/d' | sed '/^trec-car-tools @/d')"
# ^ we remove problematic lines to have a working pip-audit. See https://github.com/pypa/pip-audit/issues/84#issuecomment-1326203111 for "requests"
# ^ we also ignore GHSA-47fc-vmwq-366v vulnerability because it has no fix for the moment
