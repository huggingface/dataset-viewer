MATURIN := $(shell command -v maturin 2>/dev/null)
ifndef MATURIN
$(error maturin not found in PATH. Please install it first.)
endif

.PHONY: install
install:
	$(MATURIN) develop -r