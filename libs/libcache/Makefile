include ../../tools/Common.mk

.PHONY: test
test:
	MONGO_CACHE_DATABASE="datasets_server_cache_test" poetry run python -m pytest -x tests

.PHONY: coverage
coverage:
	MONGO_CACHE_DATABASE="datasets_server_cache_test" poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests
