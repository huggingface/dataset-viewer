include ../../tools/Common.mk

.PHONY: test
test:
	docker-compose -f tests/docker-compose.yml up -d --remove-orphans
	MONGO_URL="mongodb://localhost:27020" MONGO_QUEUE_DATABASE="datasets_server_queue_test" poetry run python -m pytest -x tests
	docker-compose -f tests/docker-compose.yml down

.PHONY: coverage
coverage:
	docker-compose -f tests/docker-compose.yml up -d --remove-orphans
	MONGO_URL="mongodb://localhost:27020" MONGO_QUEUE_DATABASE="datasets_server_queue_test" poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests
	docker-compose -f tests/docker-compose.yml down
