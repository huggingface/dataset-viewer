.PHONY: cancel-jobs-splits-next
cancel-jobs-splits-next:
	poetry run python src/admin/scripts/cancel_jobs_splits_next.py

.PHONY: cancel-jobs-first-rows
cancel-jobs-first-rows:
	poetry run python src/admin/scripts/cancel_jobs_first_rows.py

.PHONY: refresh-cache
refresh-cache:
	poetry run python src/admin/scripts/refresh_cache.py

.PHONY: refresh-cache-canonical
refresh-cache-canonical:
	poetry run python src/admin/scripts/refresh_cache_canonical.py

.PHONY: refresh-cache-errors
refresh-cache-errors:
	poetry run python src/admin/scripts/refresh_cache_errors.py
