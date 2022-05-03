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
	ROWS_MAX_NUMBER=5 MONGO_CACHE_DATABASE="datasets_preview_cache_test" poetry run python -m pytest -x tests

# Ensure to specify HF_TOKEN when calling make coverage, ie HF_TOKEN=hf_app_xxx make coverage
.PHONY: coverage
coverage:
	ROWS_MAX_NUMBER=5 MONGO_CACHE_DATABASE="datasets_preview_cache_test" poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests
