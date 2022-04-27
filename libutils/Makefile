include ../tools/Common.mk

.PHONY: test
test:
	poetry run python -m pytest -x tests

.PHONY: coverage
coverage:
	poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests
