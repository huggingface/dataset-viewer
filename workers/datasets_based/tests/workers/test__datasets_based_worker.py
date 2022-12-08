# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import datasets.config
import pytest

from datasets_based.config import AppConfig
from datasets_based.workers._datasets_based_worker import DatasetsBasedWorker


class DummyWorker(DatasetsBasedWorker):
    @staticmethod
    def get_endpoint() -> str:
        return "/splits"
        # ^ borrowing the endpoint, so that the processing step exists and the worker can be initialized
        # refactoring libcommon.processing_graph might help avoiding this

    def compute(
        self, dataset: str, config: Optional[str] = None, split: Optional[str] = None, force: bool = False
    ) -> Mapping[str, Any]:
        if config == "raise":
            raise ValueError("This is a test")
        else:
            return {}


@pytest.fixture
def worker(app_config: AppConfig) -> DummyWorker:
    return DummyWorker(app_config=app_config)


def test_version(worker: DummyWorker) -> None:
    assert len(worker.version.split(".")) == 3
    assert worker.compare_major_version(other_version="0.0.0") > 0
    assert worker.compare_major_version(other_version="1000.0.0") < 0


def test_has_storage(worker: DummyWorker) -> None:
    assert worker.has_storage() is True
    worker.datasets_based_config.max_disk_usage_percent = 0
    # the directory does not exist yet, so it should return True
    assert worker.has_storage() is True
    os.makedirs(worker.datasets_based_config.hf_datasets_cache, exist_ok=True)
    assert worker.has_storage() is False


@pytest.mark.parametrize(
    "dataset,config,split,force,expected",
    [
        ("user/dataset", "config", "split", True, "2022-11-07-12-34-56--splits-user-dataset-775e7212"),
        # Every parameter variation changes the hash, hence the subdirectory
        ("user/dataset", None, "split", True, "2022-11-07-12-34-56--splits-user-dataset-73c4b810"),
        ("user/dataset", "config2", "split", True, "2022-11-07-12-34-56--splits-user-dataset-b6920bfb"),
        ("user/dataset", "config", None, True, "2022-11-07-12-34-56--splits-user-dataset-36d21623"),
        ("user/dataset", "config", "split2", True, "2022-11-07-12-34-56--splits-user-dataset-f60adde1"),
        ("user/dataset", "config", "split", False, "2022-11-07-12-34-56--splits-user-dataset-f7985698"),
        # The subdirectory length is truncated, and it always finishes with the hash
        (
            "very_long_dataset_name_0123456789012345678901234567890123456789012345678901234567890123456789",
            "config",
            "split",
            True,
            "2022-11-07-12-34-56--splits-very_long_dataset_name_0123456789012-1457d125",
        ),
    ],
)
def test_get_cache_subdirectory(
    worker: DummyWorker, dataset: str, config: Optional[str], split: Optional[str], force: bool, expected: str
) -> None:
    date = datetime(2022, 11, 7, 12, 34, 56)
    subdirectory = worker.get_cache_subdirectory(date=date, dataset=dataset, config=config, split=split, force=force)
    assert subdirectory == expected


def test_set_and_unset_datasets_cache(worker: DummyWorker) -> None:
    base_path = worker.datasets_based_config.hf_datasets_cache
    dummy_path = base_path / "dummy"
    worker.set_datasets_cache(dummy_path)
    assert_datasets_cache_path(path=dummy_path, exists=True)
    worker.unset_datasets_cache()
    assert_datasets_cache_path(path=base_path, exists=True)


def test_set_and_unset_modules_cache(worker: DummyWorker) -> None:
    base_path = worker.datasets_based_config.hf_modules_cache
    dummy_path = base_path / "dummy"
    worker.set_modules_cache(dummy_path)
    assert_modules_cache_path(path=dummy_path, exists=True)
    worker.unset_modules_cache()
    assert_modules_cache_path(path=base_path, exists=True)


def test_set_and_unset_cache(worker: DummyWorker) -> None:
    datasets_base_path = worker.datasets_based_config.hf_datasets_cache
    modules_base_path = worker.datasets_based_config.hf_modules_cache
    worker.set_cache(dataset="user/dataset", config="config", split="split", force=True)
    assert str(datasets.config.HF_DATASETS_CACHE).startswith(str(datasets_base_path))
    assert "-splits-user-dataset" in str(datasets.config.HF_DATASETS_CACHE)
    worker.unset_cache()
    assert_datasets_cache_path(path=datasets_base_path, exists=True)
    assert_modules_cache_path(path=modules_base_path, exists=True)


@pytest.mark.parametrize("config", ["raise", "dont_raise"])
def test_process(worker: DummyWorker, hub_public_csv: str, config: str) -> None:
    # ^ this test requires an existing dataset, otherwise .process fails before setting the cache
    # it must work in both cases: when the job fails and when it succeeds
    datasets_base_path = worker.datasets_based_config.hf_datasets_cache
    modules_base_path = worker.datasets_based_config.hf_modules_cache
    # the datasets library sets the cache to its own default
    assert_datasets_cache_path(path=datasets_base_path, exists=False, equals=False)
    assert_modules_cache_path(path=modules_base_path, exists=False, equals=False)
    result = worker.process(dataset=hub_public_csv, config=config, force=True)
    assert result is (config != "raise")
    # the configured cache is now set (after having deleted a subdirectory used for the job)
    assert_datasets_cache_path(path=datasets_base_path, exists=True)
    assert_modules_cache_path(path=modules_base_path, exists=True)


def assert_datasets_cache_path(path: Path, exists: bool, equals: bool = True) -> None:
    assert path.exists() is exists
    assert (datasets.config.HF_DATASETS_CACHE == path) is equals
    assert (datasets.config.DOWNLOADED_DATASETS_PATH == path / datasets.config.DOWNLOADED_DATASETS_DIR) is equals
    assert (datasets.config.EXTRACTED_DATASETS_PATH == path / datasets.config.EXTRACTED_DATASETS_DIR) is equals


def assert_modules_cache_path(path: Path, exists: bool, equals: bool = True) -> None:
    assert path.exists() is exists
    assert (datasets.config.HF_MODULES_CACHE == path) is equals
