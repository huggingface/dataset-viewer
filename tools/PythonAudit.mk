.PHONY: pip-audit
pip-audit:
	bash -c 'poetry run pip-audit -r <(poetry export -f requirements.txt --with dev)'
