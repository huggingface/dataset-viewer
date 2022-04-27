include ../../tools/Common.mk

.PHONY: test
test:
	MONGO_QUEUE_DATABASE="datasets_preview_queue_test" poetry run python -m pytest -x tests

.PHONY: coverage
coverage:
	MONGO_QUEUE_DATABASE="datasets_preview_queue_test" poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests
