
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
	poetry run ruff check src
	poetry run ruff check tests --ignore=ARG
	poetry run ruff format --check src tests
	poetry run mypy tests src
	poetry run bandit -r src
	$(MAKE) pip-audit

# Format source code automatically
.PHONY: style
style:
	poetry run ruff check --fix src
	poetry run ruff check --fix tests --ignore=ARG
	poetry run ruff format src tests

.PHONY: pip-audit
pip-audit:
	bash -c "poetry run pip-audit --ignore-vuln GHSA-wfm5-v35h-vwf4 --ignore-vuln GHSA-cwvm-v4w8-q58c -r <(poetry export -f requirements.txt --with dev  | sed '/^kenlm @/d' |sed '/^torch @/d' | sed '/^libapi @/d' | sed '/^libcommon @/d' | sed '/^trec-car-tools @/d')"
# ^ we remove problematic lines to have a working pip-audit. See https://github.com/pypa/pip-audit/issues/84#issuecomment-1326203111 for "requests"
