
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
	$(MAKE) pip-audit

# Format source code automatically
.PHONY: style
style:
	poetry run black tests src
	poetry run isort tests src

.PHONY: pip-audit
pip-audit:
	bash -c "poetry run pip-audit -r <(poetry export -f requirements.txt --with dev  | sed '/^pymongo==/,+109 d' | sed '/^requests==2.28.2 ;/,+2 d' | sed '/^kenlm @/d' | sed '/^fsspec==/,+2 d' | sed '/^torch @/d' | sed '/^torchaudio @/d' | sed '/^libcommon @/d' | sed '/^trec-car-tools @/d' | sed '/^hffs @/d')"
# ^ we remove problematic lines to have a working pip-audit. See https://github.com/pypa/pip-audit/issues/84#issuecomment-1326203111 for "requests"
