# monitor the load with htop and adapt the value (load should be a bit less than the number of processors (check with "$ nproc"))
# this will launch as much processes as possible under the limit of load=MAX_LOAD
MAX_LOAD = 7
PARALLEL = -j -l $(MAX_LOAD)

.PHONY: install run test quality style benchmark watch

install:
	poetry install


run:
	poetry run python src/datasets_preview_backend/main.py

watch:
	poetry run watchmedo auto-restart -d src/datasets_preview_backend -p "*.py" -R python src/datasets_preview_backend/main.py

test:
	poetry run python -m pytest -x tests

# Check that source code meets quality standards + security
quality:
	poetry run black --check tests src benchmark
	poetry run isort --check-only tests src benchmark
	poetry run flake8 tests src benchmark
	poetry run bandit -r src benchmark
	poetry run safety check -i 41161

# Format source code automatically
style:
	poetry run black tests src benchmark
	poetry run isort tests src benchmark

# Get a report for every dataset / config / split of the Hub, for every endpoint
# It takes 1 or 2 hours to run. Delete benchmark/tmp to run from scratch.
# beware: even if all the data should theoritically be streamed, the ~/.cache/huggingface directory
# will grow about 25G!
# The result is benchmark/tmp/report.json (about 40M)
benchmark:
	$(MAKE) -C benchmark $(PARALLEL)
