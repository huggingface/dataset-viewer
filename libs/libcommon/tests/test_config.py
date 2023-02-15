# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.config import CommonConfig


def test_common_config() -> None:
    common_config = CommonConfig()
    assert common_config.log_level == 20
