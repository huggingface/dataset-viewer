include ../../tools/Common.mk

.PHONY: cancel-started-split-jobs
cancel-started-split-jobs:
	poetry run python src/admin/scripts/cancel_started_split_jobs.py

.PHONY: cancel-started-dataset-jobs
cancel-started-dataset-jobs:
	poetry run python src/admin/scripts/cancel_started_dataset_jobs.py

.PHONY: warm-cache
warm-cache:
	poetry run python src/admin/scripts/warm_cache.py

# Ensure to specify HF_TOKEN when calling make test, ie HF_TOKEN=hf_app_xxx make test
.PHONY: test
test:
	docker-compose -f tests/docker-compose.yml up -d
	MONGO_CACHE_DATABASE="datasets_server_cache_test" MONGO_QUEUE_DATABASE="datasets_server_queue_test" MONGO_URL="mongodb://localhost:27019" poetry run python -m pytest -x tests
	docker-compose -f tests/docker-compose.yml down

# Ensure to specify HF_TOKEN when calling make coverage, ie HF_TOKEN=hf_app_xxx make coverage
.PHONY: coverage
coverage:
	docker-compose -f tests/docker-compose.yml up -d
	MONGO_CACHE_DATABASE="datasets_server_cache_test" MONGO_QUEUE_DATABASE="datasets_server_queue_test" MONGO_URL="mongodb://localhost:27019" poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests
	docker-compose -f tests/docker-compose.yml down
