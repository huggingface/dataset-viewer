include ../../tools/Common.mk

.PHONY: datasets-worker
datasets-worker:
	WORKER_QUEUE=dataset poetry run python src/job_runner/main.py

.PHONY: splits-worker
splits-worker:
	WORKER_QUEUE=splits poetry run python src/job_runner/main.py

.PHONY: test
test:
	ROWS_MAX_NUMBER=5 MONGO_CACHE_DATABASE="datasets_preview_cache_test" poetry run python -m pytest -x tests

.PHONY: coverage
coverage:
	ROWS_MAX_NUMBER=5 MONGO_CACHE_DATABASE="datasets_preview_cache_test" poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests
