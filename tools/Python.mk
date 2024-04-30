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
# Run ruff linter
	if [ -d src ]; then $(POETRY) run ruff check src; fi
	if [ -d tests ]; then $(POETRY) run ruff check tests --ignore=ARG; fi
# Run ruff formatter
	if [ -d src ]; then $(POETRY) run ruff format --check src; fi
	if [ -d tests ]; then $(POETRY) run ruff format --check tests; fi
# Run mypy
	if [ -d src ]; then $(POETRY) run mypy src; fi
	if [ -d tests ]; then $(POETRY) run mypy tests; fi
# Run bandit
	if [ -d src ]; then $(POETRY) run bandit -r src; fi

# Format source code automatically
.PHONY: style
style:
# Run ruff linter
	if [ -d src ]; then $(POETRY) run ruff check --fix src; fi
	if [ -d tests ]; then $(POETRY) run ruff check --fix tests --ignore=ARG; fi
# Run ruff formatter
	if [ -d src ]; then $(POETRY) run ruff format src; fi
	if [ -d tests ]; then $(POETRY) run ruff format tests; fi

.PHONY: pip-audit
pip-audit:
	bash -c "$(POETRY) run pip-audit --ignore-vuln GHSA-wj6h-64fc-37mp --ignore-vuln GHSA-wfm5-v35h-vwf4 --ignore-vuln GHSA-cwvm-v4w8-q58c --ignore-vuln PYSEC-2022-43059 -r <($(POETRY) export -f requirements.txt --with dev  | sed '/^libapi @/d' | sed '/^libcommon @/d')"
