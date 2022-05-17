
.PHONY: install
install:
	$(MAKE) -C services/worker/ install
	$(MAKE) -C services/api/ install
	$(MAKE) -C services/admin/ install

.PHONY: lock
lock:
	$(MAKE) -C libs/libutils/ lock
	$(MAKE) -C libs/libqueue/ lock
	$(MAKE) -C libs/libcache/ lock
	$(MAKE) -C services/worker/ lock
	$(MAKE) -C services/api/ lock
	$(MAKE) -C services/admin/ lock

.PHONY: api
api:
	$(MAKE) -C services/api/ run

.PHONY: worker
worker:
	$(MAKE) -C services/worker/ run

.PHONY: test
test:
	$(MAKE) -C services/admin/ test
	$(MAKE) -C services/worker/ test
	$(MAKE) -C services/api/ test
	$(MAKE) -C libs/libcache/ test
	$(MAKE) -C libs/libqueue/ test
	$(MAKE) -C libs/libutils/ test

.PHONY: coverage
coverage:
	$(MAKE) -C services/admin/ coverage
	$(MAKE) -C services/worker/ coverage
	$(MAKE) -C services/api/ coverage
	$(MAKE) -C libs/libcache/ coverage
	$(MAKE) -C libs/libqueue/ coverage
	$(MAKE) -C libs/libutils/ coverage

# Check that source code meets quality standards + security
.PHONY: quality
quality:
	$(MAKE) -C e2e/ quality
	$(MAKE) -C infra/charts/datasets-server/ quality
	$(MAKE) -C services/worker/ quality
	$(MAKE) -C services/api/ quality
	$(MAKE) -C services/admin/ quality
	$(MAKE) -C libs/libcache/ quality
	$(MAKE) -C libs/libqueue/ quality
	$(MAKE) -C libs/libutils/ quality

# Format source code automatically
.PHONY: style
style:
	$(MAKE) -C e2e/ style
	$(MAKE) -C services/worker/ style
	$(MAKE) -C services/api/ style
	$(MAKE) -C services/admin/ style
	$(MAKE) -C libs/libcache/ style
	$(MAKE) -C libs/libqueue/ style
	$(MAKE) -C libs/libutils/ style

.PHONY: e2e
e2e:
	$(MAKE) -C e2e/ e2e
