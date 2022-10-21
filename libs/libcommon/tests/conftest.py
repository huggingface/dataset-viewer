# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from libcommon.config import CommonConfig


@pytest.fixture(scope="session")
def common_config():
    return CommonConfig()
