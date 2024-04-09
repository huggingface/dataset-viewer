BUILD_DIR?=~/tmp/doc-datasets-server
POETRY := $(shell command -v poetry@1.7.1 2> /dev/null)
POETRY_DEFAULT := $(shell command -v poetry 2> /dev/null)
POETRY := $(if $(POETRY),$(POETRY),$(POETRY_DEFAULT))

.PHONY: install
install:
	$(POETRY) install --no-root

.PHONY: build
build:
	$(POETRY) run doc-builder build datasets-server source/ --build_dir $(BUILD_DIR) --not_python_module

.PHONY: preview
preview:
	$(POETRY) run doc-builder preview datasets-server source/ --not_python_module
