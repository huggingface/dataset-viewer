BUILD_DIR?=~/tmp/doc-datasets-server

.PHONY: install
install:
	poetry install

.PHONY: build
build:
	poetry run doc-builder build datasets-server source/ --build_dir $(BUILD_DIR) --not_python_module
