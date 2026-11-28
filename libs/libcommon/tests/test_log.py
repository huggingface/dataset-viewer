# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import logging
from collections.abc import Iterator

import pytest

from libcommon.log import init_logging


@pytest.fixture
def preserve_root_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    try:
        yield
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


def test_init_logging_overrides_a_preexisting_root_handler(preserve_root_logging: None) -> None:
    # A dependency imported before init_logging() (datasets, torch, ...) may already have attached a
    # handler to the root logger. Without force=True, logging.basicConfig() is then a no-op, the root
    # logger stays at its default WARNING level, and every root INFO/DEBUG log is silently dropped.
    root = logging.getLogger()
    root.handlers[:] = [logging.NullHandler()]
    root.setLevel(logging.WARNING)

    init_logging(level=logging.INFO)

    assert root.getEffectiveLevel() == logging.INFO
    assert root.isEnabledFor(logging.INFO)


@pytest.mark.parametrize("logger_name", ["botocore", "botocore.endpoint", "botocore.auth", "aiobotocore", "s3fs"])
def test_init_logging_keeps_aws_clients_quiet_at_debug_level(preserve_root_logging: None, logger_name: str) -> None:
    init_logging(level=logging.DEBUG)

    assert logging.getLogger().isEnabledFor(logging.DEBUG)
    assert not logging.getLogger(logger_name).isEnabledFor(logging.INFO)
    assert logging.getLogger(logger_name).isEnabledFor(logging.WARNING)
