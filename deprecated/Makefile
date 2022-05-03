
.PHONY: warm
warm:
	poetry run python src/datasets_server/warm.py

.PHONY: worker
worker:
	poetry run python src/datasets_server/worker.py

.PHONY: force-refresh-cache
force-refresh-cache:
	poetry run python src/datasets_server/force_refresh_cache.py

.PHONY: cancel-started-jobs
cancel-started-jobs:
	poetry run python src/datasets_server/cancel_started_jobs.py

.PHONY: cancel-waiting-jobs
cancel-waiting-jobs:
	poetry run python src/datasets_server/cancel_waiting_jobs.py

.PHONY: clean-queues
clean-queues:
	poetry run python src/datasets_server/clean_queues.py

.PHONY: clean-cache
clean-cache:
	poetry run python src/datasets_server/clean_cache.py
# TODO: remove the assets too

.PHONY: clean
clean: clean-queues clean-cache
