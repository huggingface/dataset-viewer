POETRY := $(shell command -v poetry@1.7.1 2> /dev/null)
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

#$(MAKE) pip-audit
# ^ 20231121 - disabled until we upgrade to huggingface-hub@0.20

# Format source code automatically
.PHONY: style
style:
	$(POETRY) run ruff check --fix src
	$(POETRY) run ruff check --fix tests --ignore=ARG
	$(POETRY) run ruff format src tests
	$(POETRY) run mypy tests src
	$(POETRY) run bandit -r src

.PHONY: pip-audit
pip-audit:
	bash -c "$(POETRY) run pip-audit --ignore-vuln GHSA-wfm5-v35h-vwf4 --ignore-vuln GHSA-cwvm-v4w8-q58c --ignore-vuln PYSEC-2022-43059 -r <(poetry export -f requirements.txt --with dev  | sed '/^kenlm @/d' |sed '/^torch @/d' | sed '/^libapi @/d' | sed '/^libcommon @/d' | sed '/^trec-car-tools @/d')"
# ^ we remove problematic lines to have a working pip-audit. See https://github.com/pypa/pip-audit/issues/84#issuecomment-1326203111 for "requests"
