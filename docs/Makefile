BUILD_DIR?=~/tmp/doc-dataset-viewer
POETRY := $(shell command -v poetry@1.8.2 2> /dev/null)
POETRY_DEFAULT := $(shell command -v poetry 2> /dev/null)
POETRY := $(if $(POETRY),$(POETRY),$(POETRY_DEFAULT))

.PHONY: install
install:
	$(POETRY) install --no-root

.PHONY: build
build:
	$(POETRY) run doc-builder build dataset-viewer source/ --build_dir $(BUILD_DIR) --not_python_module

.PHONY: preview
preview:
	$(POETRY) run doc-builder preview dataset-viewer source/ --not_python_module
