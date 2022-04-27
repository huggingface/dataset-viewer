include ../tools/Common.mk

.PHONY: test
test:
	ROWS_MAX_NUMBER=5 poetry run python -m pytest -x tests

.PHONY: coverage
coverage:
	ROWS_MAX_NUMBER=5 poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests
