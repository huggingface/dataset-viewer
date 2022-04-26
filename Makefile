
.PHONY: install
install:
	$(MAKE) -C datasets_preview_backend/ install

.PHONY: run
run:
	$(MAKE) -C datasets_preview_backend/ run

.PHONY: watch
watch:
	$(MAKE) -C datasets_preview_backend/ watch
	
.PHONY: test
test:
	$(MAKE) -C datasets_preview_backend/ test

.PHONY: coverage
coverage:
	$(MAKE) -C datasets_preview_backend/ coverage

# Check that source code meets quality standards + security
.PHONY: quality
quality:
	$(MAKE) -C datasets_preview_backend/ quality

# Format source code automatically
.PHONY: style
style:
	$(MAKE) -C datasets_preview_backend/ style

.PHONY: warm
warm:
	$(MAKE) -C datasets_preview_backend/ warm

.PHONY: worker
worker:
	$(MAKE) -C datasets_preview_backend/ worker

.PHONY: force-refresh-cache
force-refresh-cache:
	$(MAKE) -C datasets_preview_backend/ force-refresh-cache

.PHONY: cancel-started-jobs
cancel-started-jobs:
	$(MAKE) -C datasets_preview_backend/ cancel-started-jobs

.PHONY: cancel-waiting-jobs
cancel-waiting-jobs:
	$(MAKE) -C datasets_preview_backend/ cancel-waiting-jobs

.PHONY: clean-queues
clean-queues:
	$(MAKE) -C datasets_preview_backend/ clean-queues

.PHONY: clean-cache
clean-cache:
	$(MAKE) -C datasets_preview_backend/ clean-cache
# TODO: remove the assets too

.PHONY: clean
clean: clean-queues clean-cache
