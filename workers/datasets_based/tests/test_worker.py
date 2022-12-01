# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Optional

import pytest

from datasets_based.config import AppConfig
from datasets_based.worker import get_worker


@pytest.mark.parametrize(
    "endpoint,expected_worker",
    [
        (None, "SplitsWorker"),
        ("/splits", "SplitsWorker"),
        ("/first-rows", "SplitsWorker"),
        ("/unknown", None),
    ],
)
def test_get_worker(app_config: AppConfig, endpoint: Optional[str], expected_worker: Optional[str]) -> None:
    if endpoint is not None:
        app_config.datasets_based.endpoint = endpoint
    if expected_worker is None:
        with pytest.raises(ValueError):
            get_worker(app_config)
    else:
        worker = get_worker(app_config)
        worker.__class__.__name__ == expected_worker
