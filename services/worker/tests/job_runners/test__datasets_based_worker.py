# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import datasets.config
import pytest
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.utils import Priority

from worker.config import AppConfig
from worker.job_runners._datasets_based_job_runner import DatasetsBasedJobRunner
from worker.resources import LibrariesResource
from worker.utils import CompleteJobResult

from ..fixtures.hub import get_default_config_split


class DummyJobRunner(DatasetsBasedJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "/config-names"
        # ^ borrowing the type, so that the processing step exists and the job runner can be initialized
        # refactoring libcommon.processing_graph might help avoiding this

    @staticmethod
    def get_job_runner_version() -> int:
        return 1

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult({"col1": "a" * 200})


GetJobRunner = Callable[[str, Optional[str], Optional[str], AppConfig, bool], DummyJobRunner]


@pytest.fixture
def get_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: Optional[str],
        split: Optional[str],
        app_config: AppConfig,
        force: bool,
    ) -> DummyJobRunner:
        processing_step_name = DummyJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            {
                processing_step_name: {
                    "input_type": "dataset",
                    "job_runner_version": DummyJobRunner.get_job_runner_version(),
                }
            }
        )
        return DummyJobRunner(
            job_info={
                "type": DummyJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                },
                "job_id": "job_id",
                "force": force,
                "priority": Priority.NORMAL,
            },
            app_config=app_config,
            processing_step=processing_graph.get_processing_step(processing_step_name),
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,config,split,force,expected",
    [
        ("user/dataset", "config", "split", True, "2022-11-07-12-34-56--config-names-user-dataset-cdf8effa"),
        # Every parameter variation changes the hash, hence the subdirectory
        ("user/dataset", None, "split", True, "2022-11-07-12-34-56--config-names-user-dataset-54ba8b96"),
        ("user/dataset", "config2", "split", True, "2022-11-07-12-34-56--config-names-user-dataset-1ad0bdcb"),
        ("user/dataset", "config", None, True, "2022-11-07-12-34-56--config-names-user-dataset-49c90a57"),
        ("user/dataset", "config", "split2", True, "2022-11-07-12-34-56--config-names-user-dataset-9a5cd356"),
        ("user/dataset", "config", "split", False, "2022-11-07-12-34-56--config-names-user-dataset-abec311a"),
        # The subdirectory length is truncated, and it always finishes with the hash
        (
            "very_long_dataset_name_0123456789012345678901234567890123456789012345678901234567890123456789",
            "config",
            "split",
            True,
            "2022-11-07-12-34-56--config-names-very_long_dataset_name_0123456-30acf104",
        ),
    ],
)
def test_get_cache_subdirectory(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    config: Optional[str],
    split: Optional[str],
    force: bool,
    expected: str,
) -> None:
    date = datetime(2022, 11, 7, 12, 34, 56)
    job_runner = get_job_runner(dataset, config, split, app_config, force)
    assert job_runner.get_cache_subdirectory(date=date) == expected


def test_set_and_unset_datasets_cache(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset, config, split = get_default_config_split("dataset")
    job_runner = get_job_runner(dataset, config, split, app_config, False)
    base_path = job_runner.base_datasets_cache
    dummy_path = base_path / "dummy"
    job_runner.set_datasets_cache(dummy_path)
    assert_datasets_cache_path(path=dummy_path, exists=True)
    job_runner.unset_datasets_cache()
    assert_datasets_cache_path(path=base_path, exists=True)


def test_set_and_unset_cache(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset, config, split = get_default_config_split("user/dataset")
    job_runner = get_job_runner(dataset, config, split, app_config, False)
    datasets_base_path = job_runner.base_datasets_cache
    job_runner.set_cache()
    assert str(datasets.config.HF_DATASETS_CACHE).startswith(str(datasets_base_path))
    assert "-config-names-user-dataset" in str(datasets.config.HF_DATASETS_CACHE)
    job_runner.unset_cache()
    assert_datasets_cache_path(path=datasets_base_path, exists=True)


def assert_datasets_cache_path(path: Path, exists: bool, equals: bool = True) -> None:
    assert path.exists() is exists
    assert (datasets.config.HF_DATASETS_CACHE == path) is equals
    assert (datasets.config.DOWNLOADED_DATASETS_PATH == path / datasets.config.DOWNLOADED_DATASETS_DIR) is equals
    assert (datasets.config.EXTRACTED_DATASETS_PATH == path / datasets.config.EXTRACTED_DATASETS_DIR) is equals
