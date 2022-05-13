include ../../tools/Common.mk

.PHONY: run
run:
	poetry run python src/api/main.py

.PHONY: watch
watch:
	poetry run watchmedo auto-restart -d src/api -p "*.py" -R python src/api/main.py

.PHONY: test
test:
	docker-compose -f tests/docker-compose.yml up -d
	MONGO_CACHE_DATABASE="datasets_server_cache_test" MONGO_QUEUE_DATABASE="datasets_server_queue_test" MONGO_URL="mongodb://localhost:27018" poetry run python -m pytest -x tests
	docker-compose -f tests/docker-compose.yml down

.PHONY: coverage
coverage:
	docker-compose -f tests/docker-compose.yml up -d
	MONGO_CACHE_DATABASE="datasets_server_cache_test" MONGO_QUEUE_DATABASE="datasets_server_queue_test" MONGO_URL="mongodb://localhost:27018" poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests
	docker-compose -f tests/docker-compose.yml down
