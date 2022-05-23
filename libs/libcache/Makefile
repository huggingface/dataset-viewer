include ../../tools/Common.mk

.PHONY: test
test:
	docker-compose -f tests/docker-compose.yml up -d
	MONGO_CACHE_DATABASE="datasets_server_cache_test" MONGO_URL="mongodb://localhost:27018" poetry run python -m pytest -x tests
	docker-compose -f tests/docker-compose.yml down

.PHONY: coverage
coverage:
	docker-compose -f tests/docker-compose.yml up -d
	MONGO_CACHE_DATABASE="datasets_server_cache_test" MONGO_URL="mongodb://localhost:27018" poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests
	docker-compose -f tests/docker-compose.yml down
