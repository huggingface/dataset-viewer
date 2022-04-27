
.PHONY: install
install:
	poetry install

.PHONY: test
test:
	poetry run python -m pytest -x tests

.PHONY: coverage
coverage:
	poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests

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
