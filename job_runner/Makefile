include ../tools/Common.mk

.PHONY: run
run:
	poetry run python src/job_runner/main.py

.PHONY: watch
watch:
	poetry run watchmedo auto-restart -d src/job_runner -p "*.py" -R python src/job_runner/main.py

.PHONY: test
test:
	MONGO_CACHE_DATABASE="datasets_preview_cache_test" MONGO_QUEUE_DATABASE="datasets_preview_queue_test" poetry run python -m pytest -x tests

.PHONY: coverage
coverage:
	MONGO_CACHE_DATABASE="datasets_preview_cache_test" MONGO_QUEUE_DATABASE="datasets_preview_queue_test" poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests
