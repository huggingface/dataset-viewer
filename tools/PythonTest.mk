TEST_PATH ?= tests

.PHONY: test
test:
	$(MAKE) up
	poetry run python -m pytest -vv -x ${ADDOPTS} $(TEST_PATH)
	$(MAKE) down

.PHONY: debug
debug:
	$(MAKE) up
	poetry run python -m pytest -vv -x --log-cli-level=DEBUG --capture=tee-sys --pdb ${ADDOPTS} $(TEST_PATH)
	$(MAKE) down

.PHONY: coverage
coverage:
	$(MAKE) up
	poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term $(TEST_PATH)
	$(MAKE) down
