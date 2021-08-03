PORT ?= 8000

.PHONY: install run test quality watch

install:
	poetry install

run:
	poetry run uvicorn --port $(PORT) --factory datasets_preview_backend.main:app

test:
	poetry run python -m pytest -x tests

quality:
	poetry run python quality/test_datasets.py

watch:
	poetry run uvicorn --port $(PORT) --factory --reload datasets_preview_backend.main:app
