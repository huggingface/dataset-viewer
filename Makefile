# monitor the load with htop and adapt the value (load should be a bit less than the number of processors (check with "$ nproc"))
# this will launch as much processes as possible under the limit of load=MAX_LOAD
MAX_LOAD = 7
PARALLEL = -j -l $(MAX_LOAD)

.PHONY: install run watch test coverage quality style warm

install:
	poetry install


run:
	poetry run python src/datasets_preview_backend/main.py

watch:
	poetry run watchmedo auto-restart -d src/datasets_preview_backend -p "*.py" -R python src/datasets_preview_backend/main.py

test:
	MONGO_CACHE_DATABASE="datasets_preview_cache_test" poetry run python -m pytest -x tests

coverage:
	MONGO_CACHE_DATABASE="datasets_preview_cache_test" poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests

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
