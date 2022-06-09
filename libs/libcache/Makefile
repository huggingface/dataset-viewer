include ../../tools/Common.mk
export MONGO_PORT := 27030

.PHONY: test
test:
	MONGO_PORT=${MONGO_PORT} docker-compose -f ./docker-compose.yml up -d
	MONGO_CACHE_DATABASE="datasets_server_cache_test" MONGO_URL="mongodb://localhost:${MONGO_PORT}" poetry run python -m pytest -x tests
	MONGO_PORT=${MONGO_PORT} docker-compose -f ./docker-compose.yml down

.PHONY: coverage
coverage:
	MONGO_PORT=${MONGO_PORT} docker-compose -f ./docker-compose.yml up -d
	MONGO_CACHE_DATABASE="datasets_server_cache_test" MONGO_URL="mongodb://localhost:${MONGO_PORT}" poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests
	MONGO_PORT=${MONGO_PORT} docker-compose -f ./docker-compose.yml down
