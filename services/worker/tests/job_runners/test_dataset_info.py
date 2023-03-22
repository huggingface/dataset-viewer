# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, Callable, List, Mapping, Optional, TypedDict

import pytest
from libcommon.dataset import DatasetNotFoundError
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response

from worker.config import AppConfig
from worker.job_runners.dataset_info import (
    DatasetInfoJobRunner,
    PreviousJob,
    PreviousStepFormatError,
    PreviousStepStatusError,
)


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, AppConfig, bool], DatasetInfoJobRunner]


class UpstreamResponse(TypedDict):
    kind: str
    dataset: str
    config: Optional[str]
    http_status: HTTPStatus
    content: Mapping[str, Any]


CONFIG_INFO_1 = {
    "features": {
        "image": {"_type": "Image"},
        "label": {
            "names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "_type": "ClassLabel",
        },
    },
    "splits": {
        "train": {
            "name": "train",
            "num_bytes": 17470800,
            "num_examples": 60000,
            "dataset_name": "dataset_ok",
        },
        "test": {
            "name": "test",
            "num_bytes": 2916432,
            "num_examples": 10000,
            "dataset_name": "dataset_ok",
        },
    },
    "download_checksums": {
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz": {
            "num_bytes": 9912422,
            "checksum": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
        },
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz": {
            "num_bytes": 28881,
            "checksum": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
        },
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz": {
            "num_bytes": 1648877,
            "checksum": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
        },
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz": {
            "num_bytes": 4542,
            "checksum": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
        },
    },
    "download_size": 11594722,
    "dataset_size": 20387232,
    "size_in_bytes": 31981954,
}

CONFIG_INFO_2 = {
    "features": {
        "image": {"_type": "Image"},
        "image2": {"_type": "Image"},
        "label": {
            "names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "_type": "ClassLabel",
        },
    },
    "splits": {
        "train": {
            "name": "train",
            "num_bytes": 5678,
            "num_examples": 3000,
            "dataset_name": "dataset_ok",
        },
        "test": {
            "name": "test",
            "num_bytes": 1234,
            "num_examples": 1000,
            "dataset_name": "dataset_ok",
        },
    },
    "download_checksums": {
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz": {
            "num_bytes": 9912422,
            "checksum": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
        },
    },
    "download_size": 9912422,
    "dataset_size": 6912,
    "size_in_bytes": 9919334,
}

DATASET_INFO_OK = {
    "config_1": CONFIG_INFO_1,
    "config_2": CONFIG_INFO_2,
}

UPSTREAM_RESPONSE_PARQUET_AND_DATASET_INFO: UpstreamResponse = UpstreamResponse(
    kind="/parquet-and-dataset-info",
    dataset="dataset_ok",
    config=None,
    http_status=HTTPStatus.OK,
    content={
        "parquet_files": [
            {"dataset": "dataset_ok", "config": "config_1", "split": "train", "size": 14281188},
            {"dataset": "dataset_ok", "config": "config_1", "split": "test", "size": 2383903},
            {"dataset": "dataset_ok", "config": "config_2", "split": "train", "size": 1234},
            {"dataset": "dataset_ok", "config": "config_2", "split": "train", "size": 6789},
            {"dataset": "dataset_ok", "config": "config_2", "split": "test", "size": 2383903},
        ],
        "dataset_info": DATASET_INFO_OK,
    },
)

UPSTREAM_RESPONSE_CONFIG_INFO_1: UpstreamResponse = UpstreamResponse(
    kind="config-info",
    dataset="dataset_ok",
    config="config_1",
    http_status=HTTPStatus.OK,
    content=CONFIG_INFO_1,
)

UPSTREAM_RESPONSE_CONFIG_INFO_2: UpstreamResponse = UpstreamResponse(
    kind="config-info",
    dataset="dataset_ok",
    config="config_2",
    http_status=HTTPStatus.OK,
    content=CONFIG_INFO_2,
)

EXPECTED_OK = (
    {
        "dataset_info": DATASET_INFO_OK,
        "pending": [],
        "failed": [],
    },
    1.0,
)

EXPECTED_PARTIAL = (
    {
        "dataset_info": {
            "config_1": CONFIG_INFO_1,
        },
        "pending": [
            PreviousJob(
                kind="config-size",
                dataset="dataset_partial",
                config="config_2",
                split=None,
            )
        ],
        "failed": [],
    },
    0.5,
)


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
        force: bool = False,
    ) -> DatasetInfoJobRunner:
        return DatasetInfoJobRunner(
            job_info={
                "type": DatasetInfoJobRunner.get_job_type(),
                "dataset": dataset,
                "config": None,
                "split": None,
                "job_id": "job_id",
                "force": force,
                "priority": Priority.NORMAL,
            },
            common_config=app_config.common,
            worker_config=app_config.worker,
            processing_step=ProcessingStep(
                name=DatasetInfoJobRunner.get_job_type(),
                input_type="dataset",
                requires=None,
                required_by_dataset_viewer=False,
                parent=None,
                ancestors=[],
                children=[],
                job_runner_version=DatasetInfoJobRunner.get_job_runner_version(),
            ),
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,upstream_responses,expected_error_code,expected,should_raise",
    [
        (
            "dataset_ok",
            [
                UPSTREAM_RESPONSE_PARQUET_AND_DATASET_INFO,
                UPSTREAM_RESPONSE_CONFIG_INFO_1,
                UPSTREAM_RESPONSE_CONFIG_INFO_2,
            ],
            None,
            EXPECTED_OK,
            False,
        ),
        (
            "dataset_ok",
            [UPSTREAM_RESPONSE_PARQUET_AND_DATASET_INFO, UPSTREAM_RESPONSE_CONFIG_INFO_1],
            None,
            EXPECTED_PARTIAL,
            False,
        ),
        (
            "status_error",
            [
                UpstreamResponse(
                    kind="/parquet-and-dataset-info",
                    dataset="status_error",
                    config=None,
                    http_status=HTTPStatus.NOT_FOUND,
                    content={"error": "error"},
                )
            ],
            PreviousStepStatusError.__name__,
            None,
            True,
        ),
        (
            "format_error",
            [
                UpstreamResponse(
                    kind="/parquet-and-dataset-info",
                    dataset="format_error",
                    config=None,
                    http_status=HTTPStatus.OK,
                    content={"not_dataset_info": "wrong_format"},
                )
            ],
            PreviousStepFormatError.__name__,
            None,
            True,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    upstream_responses: List[UpstreamResponse],
    expected_error_code: str,
    expected: Any,
    should_raise: bool,
) -> None:
    for upstream_response in upstream_responses:
        upsert_response(**upstream_response)
    job_runner = get_job_runner(dataset, app_config, False)
    if should_raise:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.type.__name__ == expected_error_code
    else:
        compute_result = job_runner.compute()
        assert compute_result.content, compute_result.progress == expected


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = "doesnotexist"
    job_runner = get_job_runner(dataset, app_config, False)
    with pytest.raises(DatasetNotFoundError):
        job_runner.compute()
