
.PHONY: install
install:
	$(MAKE) -C job_runner/ install
	$(MAKE) -C api_service/ install

.PHONY: lock
lock:
	$(MAKE) -C libutils/ lock
	$(MAKE) -C libqueue/ lock
	$(MAKE) -C libcache/ lock
	$(MAKE) -C libmodels/ lock
	$(MAKE) -C job_runner/ lock
	$(MAKE) -C api_service/ lock

.PHONY: api
api:
	$(MAKE) -C api_service/ run

.PHONY: worker
worker:
	$(MAKE) -C job_runner/ run

.PHONY: test
test:
	$(MAKE) -C job_runner/ test
	$(MAKE) -C api_service/ test
	$(MAKE) -C libcache/ test
	$(MAKE) -C libmodels/ test
	$(MAKE) -C libqueue/ test
	$(MAKE) -C libutils/ test

.PHONY: coverage
coverage:
	$(MAKE) -C job_runner/ coverage
	$(MAKE) -C api_service/ coverage
	$(MAKE) -C libcache/ coverage
	$(MAKE) -C libmodels/ coverage
	$(MAKE) -C libqueue/ coverage
	$(MAKE) -C libutils/ coverage

# Check that source code meets quality standards + security
.PHONY: quality
quality:
	$(MAKE) -C job_runner/ quality
	$(MAKE) -C api_service/ quality
	$(MAKE) -C libcache/ quality
	$(MAKE) -C libmodels/ quality
	$(MAKE) -C libqueue/ quality
	$(MAKE) -C libutils/ quality

# Format source code automatically
.PHONY: style
style:
	$(MAKE) -C job_runner/ style
	$(MAKE) -C api_service/ style
	$(MAKE) -C libcache/ style
	$(MAKE) -C libmodels/ style
	$(MAKE) -C libqueue/ style
	$(MAKE) -C libutils/ style

.PHONY: vscode
vscode:
	tools/update_vscode_setup.sh
