
.PHONY: install
install:
	$(MAKE) -C services/job_runner/ install
	$(MAKE) -C services/api_service/ install

.PHONY: lock
lock:
	$(MAKE) -C libs/libutils/ lock
	$(MAKE) -C libs/libqueue/ lock
	$(MAKE) -C libs/libcache/ lock
	$(MAKE) -C services/job_runner/ lock
	$(MAKE) -C services/api_service/ lock

.PHONY: api
api:
	$(MAKE) -C services/api_service/ run

.PHONY: worker
worker:
	$(MAKE) -C services/job_runner/ run

.PHONY: test
test:
	$(MAKE) -C services/job_runner/ test
	$(MAKE) -C services/api_service/ test
	$(MAKE) -C libs/libcache/ test
	$(MAKE) -C libs/libqueue/ test
	$(MAKE) -C libs/libutils/ test

.PHONY: coverage
coverage:
	$(MAKE) -C services/job_runner/ coverage
	$(MAKE) -C services/api_service/ coverage
	$(MAKE) -C libs/libcache/ coverage
	$(MAKE) -C libs/libqueue/ coverage
	$(MAKE) -C libs/libutils/ coverage

# Check that source code meets quality standards + security
.PHONY: quality
quality:
	$(MAKE) -C services/job_runner/ quality
	$(MAKE) -C services/api_service/ quality
	$(MAKE) -C libs/libcache/ quality
	$(MAKE) -C libs/libqueue/ quality
	$(MAKE) -C libs/libutils/ quality

# Format source code automatically
.PHONY: style
style:
	$(MAKE) -C services/job_runner/ style
	$(MAKE) -C services/api_service/ style
	$(MAKE) -C libs/libcache/ style
	$(MAKE) -C libs/libqueue/ style
	$(MAKE) -C libs/libutils/ style
