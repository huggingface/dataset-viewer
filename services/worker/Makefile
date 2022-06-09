# pillow <9.0.0
PILLOW_EXCEPTIONS := -i 44525 -i 44524 -i 44486 -i 44485 -i 45356 -i 44487
SAFETY_EXCEPTIONS := $(PILLOW_EXCEPTIONS)
MONGO_PORT := 27043

include ../../tools/Common.mk

.PHONY: run
run:
	poetry run python src/worker/main.py

.PHONY: datasets-worker
datasets-worker:
	WORKER_QUEUE=datasets poetry run python src/worker/main.py

.PHONY: splits-worker
splits-worker:
	WORKER_QUEUE=splits poetry run python src/worker/main.py

# Ensure to specify HF_TOKEN when calling make test, ie HF_TOKEN=hf_app_xxx make test
.PHONY: test
test:
	MONGO_PORT=${MONGO_PORT} docker-compose -f ./docker-compose.yml up -d
	ROWS_MAX_NUMBER=5 MONGO_CACHE_DATABASE="datasets_server_cache_test" MONGO_QUEUE_DATABASE="datasets_server_queue_test" MONGO_URL="mongodb://localhost:${MONGO_PORT}" poetry run python -m pytest -x tests
	MONGO_PORT=${MONGO_PORT} docker-compose -f ./docker-compose.yml down

# Ensure to specify HF_TOKEN when calling make coverage, ie HF_TOKEN=hf_app_xxx make coverage
.PHONY: coverage
coverage:
	MONGO_PORT=${MONGO_PORT} docker-compose -f ./docker-compose.yml up -d
	ROWS_MAX_NUMBER=5 MONGO_CACHE_DATABASE="datasets_server_cache_test" MONGO_QUEUE_DATABASE="datasets_server_queue_test" MONGO_URL="mongodb://localhost:${MONGO_PORT}" poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests
	MONGO_PORT=${MONGO_PORT} docker-compose -f ./docker-compose.yml down
