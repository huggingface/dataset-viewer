POETRY := $(shell command -v poetry@1.8.2 2> /dev/null)
POETRY_DEFAULT := $(shell command -v poetry 2> /dev/null)
POETRY := $(if $(POETRY),$(POETRY),$(POETRY_DEFAULT))

.PHONY: install
install:
	$(POETRY) install

.PHONY: lock
lock:
	rm -rf .venv/
	rm -f poetry.lock
	$(POETRY) lock
	$(POETRY) install

# Check that source code meets quality standards + security
.PHONY: quality
quality:
	$(POETRY) run ruff check src
	$(POETRY) run ruff check tests --ignore=ARG
	$(POETRY) run ruff format --check src tests
	$(POETRY) run mypy tests src
	$(POETRY) run bandit -r src

# Format source code automatically
.PHONY: style
style:
	$(POETRY) run ruff check --fix src
	$(POETRY) run ruff check --fix tests --ignore=ARG
	$(POETRY) run ruff format src tests

.PHONY: pip-audit
pip-audit:
	bash -c "$(POETRY) run pip-audit --ignore-vuln GHSA-wj6h-64fc-37mp --ignore-vuln GHSA-wfm5-v35h-vwf4 --ignore-vuln GHSA-cwvm-v4w8-q58c --ignore-vuln PYSEC-2022-43059 -r <($(POETRY) export -f requirements.txt --with dev  | sed '/^libapi @/d' | sed '/^libcommon @/d')"
