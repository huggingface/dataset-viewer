.PHONY: test
test:
	$(MAKE) up
	poetry run python -m pytest -vv -x tests
	$(MAKE) down

.PHONY: coverage
coverage:
	$(MAKE) up
	poetry run python -m pytest -s --cov --cov-report xml:coverage.xml --cov-report=term tests
	$(MAKE) down
