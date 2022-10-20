
.PHONY: install
install:
	poetry install

.PHONY: lock
lock:
	rm -rf .venv/
	rm -f poetry.lock
	poetry lock
	poetry install

.PHONY: build
build:
	poetry build

# Check that source code meets quality standards + security
.PHONY: quality
quality:
	poetry run black --check tests src
	poetry run isort --check-only tests src
	poetry run flake8 tests src
	poetry run mypy tests src
	poetry run bandit -r src
	poetry run safety check $(SAFETY_EXCEPTIONS)

# Format source code automatically
.PHONY: style
style:
	poetry run black tests src
	poetry run isort tests src

.PHONY: test
test:
	$(MAKE) down
	$(MAKE) up
	poetry run python -m pytest -vv -x tests
	$(MAKE) down

.PHONY: coverage
coverage:
	$(MAKE) down
	$(MAKE) up
	poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests
	$(MAKE) down
