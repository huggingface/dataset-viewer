# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

# from datetime import datetime
from typing import Any, Mapping, Optional

import pytest

from datasets_based.config import AppConfig
from datasets_based.workers._datasets_based_worker import DatasetsBasedWorker


def test__datasets_based_worker():
    assert DatasetsBasedWorker


class DummyWorker(DatasetsBasedWorker):
    @staticmethod
    def get_endpoint() -> str:
        return "/splits"
        # ^ borrowing the endpoint, so that the processing step exists and the worker can be initialized
        # refactoring libcommon.processing_graph might help avoiding this

    def compute(
        self, dataset: str, config: Optional[str] = None, split: Optional[str] = None, force: bool = False
    ) -> Mapping[str, Any]:
        return {}


@pytest.fixture
def worker(app_config: AppConfig) -> DummyWorker:
    return DummyWorker(app_config=app_config)


def test_version(worker: DummyWorker) -> None:
    assert len(worker.version.split(".")) == 3
    assert worker.compare_major_version(other_version="0.0.0") > 0
    assert worker.compare_major_version(other_version="1000.0.0") < 0


# @pytest.mark.wip
# def test_get_cache_subdirectory(worker: DummyWorker) -> None:
#     date = datetime(2022, 11, 7, 12, 34, 56)
#     dataset = "user/dataset"
#     config = "wEiRd-;:config"
#     split = "train"
#     force = True
#     subdirectory = worker.get_cache_subdirectory(
#         date=date,
#         dataset=dataset,
#         config=config,
#         split=split,
#         force=force,
#     )
#     assert subdirectory == "2022-11-07-12-34-56/user/dataset/wEiRd-;:config/train/force"
