
.PHONY: install
install:
	poetry install


.PHONY: run
run:
	poetry run python src/datasets_preview_backend/main.py

.PHONY: watch
watch:
	poetry run watchmedo auto-restart -d src/datasets_preview_backend -p "*.py" -R python src/datasets_preview_backend/main.py

.PHONY: test
test:
	EXTRACT_ROWS_LIMIT=5 MONGO_CACHE_DATABASE="datasets_preview_cache_test" MONGO_QUEUE_DATABASE="datasets_preview_queue_test" poetry run python -m pytest -x tests

.PHONY: coverage
coverage:
	EXTRACT_ROWS_LIMIT=5 MONGO_CACHE_DATABASE="datasets_preview_cache_test" MONGO_QUEUE_DATABASE="datasets_preview_queue_test" poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests

# Check that source code meets quality standards + security
.PHONY: quality
quality:
	poetry run black --check tests src
	poetry run isort --check-only tests src
	poetry run flake8 tests src
	poetry run mypy tests src
	poetry run bandit -r src
	poetry run safety check

# Format source code automatically
.PHONY: style
style:
	poetry run black tests src
	poetry run isort tests src

.PHONY: warm
warm:
	poetry run python src/datasets_preview_backend/warm.py

.PHONY: worker
worker:
	poetry run python src/datasets_preview_backend/worker.py

.PHONY: force-refresh-cache
force-refresh-cache:
	poetry run python src/datasets_preview_backend/force_refresh_cache.py

.PHONY: cancel-started-jobs
cancel-started-jobs:
	poetry run python src/datasets_preview_backend/cancel_started_jobs.py

.PHONY: cancel-waiting-jobs
cancel-waiting-jobs:
	poetry run python src/datasets_preview_backend/cancel_waiting_jobs.py

.PHONY: clean-queues
clean-queues:
	poetry run python src/datasets_preview_backend/clean_queues.py

.PHONY: clean-cache
clean-cache-all:
	poetry run python src/datasets_preview_backend/clean_cache.py

.PHONY: clean
clean: clean-queues clean-cache
