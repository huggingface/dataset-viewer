.PHONY: install
install:
	poetry install

.PHONY: lock
lock:
	rm -rf .venv/
	rm -f poetry.lock
	poetry lock
	poetry install

# Check that source code meets quality standards + security
.PHONY: quality
quality:
	poetry run black --check tests src
	poetry run isort --check-only tests src
	poetry run flake8 tests src
	poetry run mypy tests src
	poetry run bandit -r src
	poetry run safety check -i 44487 -i 44485 -i 44524 -i 44525 -i 44486 -i 44716 -i 44717 -i 44715 -i 45356 -i 46499
# ^^ safety exceptions: pillow, numpy

# Format source code automatically
.PHONY: style
style:
	poetry run black tests src
	poetry run isort tests src
