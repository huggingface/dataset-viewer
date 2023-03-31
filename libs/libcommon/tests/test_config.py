# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.config import LogConfig


def test_common_config() -> None:
    log_config = LogConfig()
    assert log_config.level == 20
