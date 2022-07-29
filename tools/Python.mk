
export TEST_MONGO_URL := mongodb://localhost:${TEST_MONGO_PORT}

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


.PHONY: test-target
test-target:
	MONGO_URL=${TEST_MONGO_URL} MONGO_QUEUE_DATABASE=${TEST_MONGO_QUEUE_DATABASE} MONGO_CACHE_DATABASE=${TEST_MONGO_CACHE_DATABASE} ROWS_MAX_NUMBER=${TEST_ROWS_MAX_NUMBER} poetry run python -m pytest -vv -x $(TEST_TARGET) $(PYTEST_ARGS)

.PHONY: test-target-expression
test-target-expression:
	MONGO_URL=${TEST_MONGO_URL} MONGO_QUEUE_DATABASE=${TEST_MONGO_QUEUE_DATABASE} MONGO_CACHE_DATABASE=${TEST_MONGO_CACHE_DATABASE} ROWS_MAX_NUMBER=${TEST_ROWS_MAX_NUMBER} poetry run python -m pytest -vv -x $(TEST_TARGET) -k $(TEST_EXPRESSION) $(PYTEST_ARGS)

.PHONY: test
test:
	MONGO_PORT=${TEST_MONGO_PORT} COMPOSE_PROJECT_NAME=${TEST_COMPOSE_PROJECT_NAME} DOCKER_COMPOSE=${TEST_DOCKER_COMPOSE} $(MAKE) down
	MONGO_PORT=${TEST_MONGO_PORT} COMPOSE_PROJECT_NAME=${TEST_COMPOSE_PROJECT_NAME} DOCKER_COMPOSE=${TEST_DOCKER_COMPOSE} ROWS_MAX_NUMBER=${TEST_ROWS_MAX_NUMBER} $(MAKE) up
	TEST_TARGET=tests make test-target
	MONGO_PORT=${TEST_MONGO_PORT} COMPOSE_PROJECT_NAME=${TEST_COMPOSE_PROJECT_NAME} DOCKER_COMPOSE=${TEST_DOCKER_COMPOSE} $(MAKE) down

.PHONY: coverage
coverage:
	MONGO_PORT=${TEST_MONGO_PORT} COMPOSE_PROJECT_NAME=${TEST_COMPOSE_PROJECT_NAME} DOCKER_COMPOSE=${TEST_DOCKER_COMPOSE} $(MAKE) down
	MONGO_PORT=${TEST_MONGO_PORT} COMPOSE_PROJECT_NAME=${TEST_COMPOSE_PROJECT_NAME} DOCKER_COMPOSE=${TEST_DOCKER_COMPOSE} ROWS_MAX_NUMBER=${TEST_ROWS_MAX_NUMBER} $(MAKE) up
	MONGO_URL=${TEST_MONGO_URL} MONGO_QUEUE_DATABASE=${TEST_MONGO_QUEUE_DATABASE} MONGO_CACHE_DATABASE=${TEST_MONGO_CACHE_DATABASE} ROWS_MAX_NUMBER=${TEST_ROWS_MAX_NUMBER} poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests
	MONGO_PORT=${TEST_MONGO_PORT} COMPOSE_PROJECT_NAME=${TEST_COMPOSE_PROJECT_NAME} DOCKER_COMPOSE=${TEST_DOCKER_COMPOSE} $(MAKE) down
