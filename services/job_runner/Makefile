
.PHONY: install
install:
	poetry install

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

# Check that source code meets quality standards + security
.PHONY: quality
quality:
	poetry run black --check tests src
	poetry run isort --check-only tests src
	poetry run flake8 tests src
	poetry run mypy tests src
	poetry run bandit -r src
	poetry run safety check -i 44487 -i 44485 -i 44524 -i 44525 -i 44486 -i 44716 -i 44717 -i 44715 -i 45356 -i 46499
# ^^ safety exceptions: pillow, numpy

# Format source code automatically
.PHONY: style
style:
	poetry run black tests src
	poetry run isort tests src
