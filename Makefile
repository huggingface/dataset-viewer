
.PHONY: install
install:
	$(MAKE) -C services/datasets_preview_backend/ install
	$(MAKE) -C services/api_service/ install

.PHONY: run
run:
	$(MAKE) -C services/api_service/ run

.PHONY: watch
watch:
	$(MAKE) -C services/api_service/ watch
	
.PHONY: test
test:
	$(MAKE) -C services/datasets_preview_backend/ test
	$(MAKE) -C services/api_service/ test

.PHONY: coverage
coverage:
	$(MAKE) -C services/datasets_preview_backend/ coverage
	$(MAKE) -C services/api_service/ coverage

# Check that source code meets quality standards + security
.PHONY: quality
quality:
	$(MAKE) -C services/datasets_preview_backend/ quality
	$(MAKE) -C services/api_service/ quality

# Format source code automatically
.PHONY: style
style:
	$(MAKE) -C services/datasets_preview_backend/ style
	$(MAKE) -C services/api_service/ style

.PHONY: warm
warm:
	$(MAKE) -C services/datasets_preview_backend/ warm

.PHONY: worker
worker:
	$(MAKE) -C services/datasets_preview_backend/ worker

.PHONY: force-refresh-cache
force-refresh-cache:
	$(MAKE) -C services/datasets_preview_backend/ force-refresh-cache

.PHONY: cancel-started-jobs
cancel-started-jobs:
	$(MAKE) -C services/datasets_preview_backend/ cancel-started-jobs

.PHONY: cancel-waiting-jobs
cancel-waiting-jobs:
	$(MAKE) -C services/datasets_preview_backend/ cancel-waiting-jobs

.PHONY: clean-queues
clean-queues:
	$(MAKE) -C services/datasets_preview_backend/ clean-queues

.PHONY: clean-cache
clean-cache:
	$(MAKE) -C services/datasets_preview_backend/ clean-cache
# TODO: remove the assets too

.PHONY: clean
clean: clean-queues clean-cache

.PHONY: vscode
vscode:
	tools/update_vscode_setup.sh
