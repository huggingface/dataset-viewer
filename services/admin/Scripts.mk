.PHONY: cancel-jobs-splits
cancel-jobs-splits:
	poetry run python src/admin/scripts/cancel_jobs_splits.py

.PHONY: cancel-jobs-first-rows
cancel-jobs-first-rows:
	poetry run python src/admin/scripts/cancel_jobs_first_rows.py
