TEST_PATH ?= tests

.PHONY: test
test:
	$(MAKE) down
	$(MAKE) up
	$(POETRY) run python -m pytest --memray -vv -x ${ADDOPTS} $(TEST_PATH)
	$(MAKE) down

.PHONY: debug
debug:
	$(MAKE) up
	$(POETRY) run python -m pytest --memray -vv -x --log-cli-level=DEBUG --capture=tee-sys --pdb ${ADDOPTS} $(TEST_PATH)
	$(MAKE) down
