PORT ?= 8000

.PHONY: install run test quality style benchmark watch

install:
	poetry install

run:
	poetry run uvicorn --port $(PORT) --factory datasets_preview_backend.main:app

test:
	poetry run python -m pytest -x tests

# Check that source code meets quality standards
quality:
	poetry run black --check tests src benchmark
	poetry run isort --check-only tests src benchmark

# Format source code automatically
style:
	poetry run black tests src benchmark
	poetry run isort tests src benchmark

benchmark:
	$(MAKE) -C benchmark
	

watch:
	poetry run uvicorn --port $(PORT) --factory --reload datasets_preview_backend.main:app
