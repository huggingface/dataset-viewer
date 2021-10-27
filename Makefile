.PHONY: install run watch test coverage quality style warm worker refresh clean clean-queue clean-cache force-finish-queue

install:
	poetry install


run:
	poetry run python src/datasets_preview_backend/main.py

watch:
	poetry run watchmedo auto-restart -d src/datasets_preview_backend -p "*.py" -R python src/datasets_preview_backend/main.py

test:
	MONGO_CACHE_DATABASE="datasets_preview_cache_test" MONGO_QUEUE_DATABASE="datasets_preview_queue_test" poetry run python -m pytest -x tests

coverage:
	MONGO_CACHE_DATABASE="datasets_preview_cache_test" MONGO_QUEUE_DATABASE="datasets_preview_queue_test" poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests

# Check that source code meets quality standards + security
quality:
	poetry run black --check tests src
	poetry run isort --check-only tests src
	poetry run flake8 tests src
	poetry run mypy tests src
	poetry run bandit -r src
	poetry run safety check -i 41161

# Format source code automatically
style:
	poetry run black tests src
	poetry run isort tests src

refresh:
	poetry run python src/datasets_preview_backend/refresh.py
warm:
	poetry run python src/datasets_preview_backend/warm.py
worker:
	poetry run python src/datasets_preview_backend/worker.py
force-finish-queue:
	poetry run python src/datasets_preview_backend/force_finish_queue.py
clean-queue:
	poetry run python src/datasets_preview_backend/clean_queue.py
clean-cache:
	poetry run python src/datasets_preview_backend/clean_cache.py
clean: clean-queue clean-cache
