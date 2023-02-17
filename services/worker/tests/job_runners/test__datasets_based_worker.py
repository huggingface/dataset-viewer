# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import replace
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import datasets.config
import pytest
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import get_response

from worker.config import AppConfig
from worker.job_runners._datasets_based_job_runner import DatasetsBasedJobRunner
from worker.resources import LibrariesResource

from ..fixtures.hub import HubDatasets, get_default_config_split


class DummyJobRunner(DatasetsBasedJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "/splits"
        # ^ borrowing the type, so that the processing step exists and the job runner can be initialized
        # refactoring libcommon.processing_graph might help avoiding this

    @staticmethod
    def get_version() -> str:
        return "1.0.0"

    def compute(self) -> Mapping[str, Any]:
        if self.config == "raise":
            raise ValueError("This is a test")
        else:
            return {"col1": "a" * 200}


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
        return DummyJobRunner(
            job_info={
                "type": DummyJobRunner.get_job_type(),
                "dataset": dataset,
                "config": config,
                "split": split,
                "job_id": "job_id",
                "force": force,
                "priority": Priority.NORMAL,
            },
            app_config=app_config,
            processing_step=ProcessingStep(
                name=DummyJobRunner.get_job_type(),
                input_type="split",
                requires=None,
                required_by_dataset_viewer=False,
                parent=None,
                ancestors=[],
                children=[],
            ),
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
        )

    return _get_job_runner


def test_version(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset, config, split = get_default_config_split("dataset")
    job_runner = get_job_runner(dataset, config, split, app_config, False)
    assert len(job_runner.get_version().split(".")) == 3
    assert job_runner.compare_major_version(other_version="0.0.0") > 0
    assert job_runner.compare_major_version(other_version="1000.0.0") < 0


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
    assert "-splits-user-dataset" in str(datasets.config.HF_DATASETS_CACHE)
    job_runner.unset_cache()
    assert_datasets_cache_path(path=datasets_base_path, exists=True)


@pytest.mark.parametrize("config", ["raise", "dont_raise"])
def test_process(app_config: AppConfig, get_job_runner: GetJobRunner, hub_public_csv: str, config: str) -> None:
    # ^ this test requires an existing dataset, otherwise .process fails before setting the cache
    # it must work in both cases: when the job fails and when it succeeds
    dataset = hub_public_csv
    split = "split"
    job_runner = get_job_runner(dataset, config, split, app_config, False)
    datasets_base_path = job_runner.base_datasets_cache
    # the datasets library sets the cache to its own default
    assert_datasets_cache_path(path=datasets_base_path, exists=False, equals=False)
    result = job_runner.process()
    assert result is (config != "raise")
    # the configured cache is now set (after having deleted a subdirectory used for the job)
    assert_datasets_cache_path(path=datasets_base_path, exists=True)


def assert_datasets_cache_path(path: Path, exists: bool, equals: bool = True) -> None:
    assert path.exists() is exists
    assert (datasets.config.HF_DATASETS_CACHE == path) is equals
    assert (datasets.config.DOWNLOADED_DATASETS_PATH == path / datasets.config.DOWNLOADED_DATASETS_DIR) is equals
    assert (datasets.config.EXTRACTED_DATASETS_PATH == path / datasets.config.EXTRACTED_DATASETS_DIR) is equals


def test_process_big_content(hub_datasets: HubDatasets, app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset, config, split = get_default_config_split(hub_datasets["big"]["name"])
    worker = get_job_runner(
        dataset, config, split, replace(app_config, worker=replace(app_config.worker, content_max_bytes=10)), False
    )

    assert not worker.process()
    cached_response = get_response(kind=worker.processing_step.cache_kind, dataset=dataset, config=config, split=split)

    assert cached_response["http_status"] == HTTPStatus.NOT_IMPLEMENTED
    assert cached_response["error_code"] == "TooBigContentError"
