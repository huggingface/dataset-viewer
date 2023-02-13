# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any

import pytest
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response

from worker.config import AppConfig
from worker.job_runners.sizes import (
    DatasetNotFoundError,
    PreviousStepFormatError,
    PreviousStepStatusError,
    SizesJobRunner,
)


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
):
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
        force: bool = False,
    ) -> SizesJobRunner:
        return SizesJobRunner(
            job_info={
                "type": SizesJobRunner.get_job_type(),
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
                endpoint=SizesJobRunner.get_job_type(),
                input_type="dataset",
                requires=None,
                required_by_dataset_viewer=False,
                parent=None,
                ancestors=[],
                children=[],
            ),
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,upstream_status,upstream_content,expected_error_code,expected_content,should_raise",
    [
        (
            "dataset_ok",
            HTTPStatus.OK,
            {
                "parquet_files": [
                    {"dataset": "dataset_ok", "config": "config_1", "split": "train", "size": 14281188},
                    {"dataset": "dataset_ok", "config": "config_1", "split": "test", "size": 2383903},
                    {"dataset": "dataset_ok", "config": "config_2", "split": "train", "size": 1234},
                    {"dataset": "dataset_ok", "config": "config_2", "split": "train", "size": 6789},
                    {"dataset": "dataset_ok", "config": "config_2", "split": "test", "size": 2383903},
                ],
                "dataset_info": {
                    "config_1": {
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
                    },
                    "config_2": {
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
                    },
                },
            },
            None,
            {
                "sizes": {
                    "dataset": {
                        "dataset": "dataset_ok",
                        "num_bytes_original_files": 21507144,
                        "num_bytes_parquet_files": 19057017,
                        "num_bytes_memory": 20394144,
                        "num_rows": 74000,
                    },
                    "configs": [
                        {
                            "dataset": "dataset_ok",
                            "config": "config_1",
                            "num_bytes_original_files": 11594722,
                            "num_bytes_parquet_files": 16665091,
                            "num_bytes_memory": 20387232,
                            "num_rows": 70000,
                            "num_columns": 2,
                        },
                        {
                            "dataset": "dataset_ok",
                            "config": "config_2",
                            "num_bytes_original_files": 9912422,
                            "num_bytes_parquet_files": 2391926,
                            "num_bytes_memory": 6912,
                            "num_rows": 4000,
                            "num_columns": 3,
                        },
                    ],
                    "splits": [
                        {
                            "dataset": "dataset_ok",
                            "config": "config_1",
                            "split": "train",
                            "num_bytes_parquet_files": 14281188,
                            "num_bytes_memory": 17470800,
                            "num_rows": 60000,
                            "num_columns": 2,
                        },
                        {
                            "dataset": "dataset_ok",
                            "config": "config_1",
                            "split": "test",
                            "num_bytes_parquet_files": 2383903,
                            "num_bytes_memory": 2916432,
                            "num_rows": 10000,
                            "num_columns": 2,
                        },
                        {
                            "dataset": "dataset_ok",
                            "config": "config_2",
                            "split": "train",
                            "num_bytes_parquet_files": 8023,
                            "num_bytes_memory": 5678,
                            "num_rows": 3000,
                            "num_columns": 3,
                        },
                        {
                            "dataset": "dataset_ok",
                            "config": "config_2",
                            "split": "test",
                            "num_bytes_parquet_files": 2383903,
                            "num_bytes_memory": 1234,
                            "num_rows": 1000,
                            "num_columns": 3,
                        },
                    ],
                }
            },
            False,
        ),
        ("status_error", HTTPStatus.NOT_FOUND, {"error": "error"}, PreviousStepStatusError.__name__, None, True),
        (
            "format_error",
            HTTPStatus.OK,
            {"not_dataset_info": "wrong_format"},
            PreviousStepFormatError.__name__,
            None,
            True,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner,
    dataset: str,
    upstream_status: HTTPStatus,
    upstream_content: Any,
    expected_error_code: str,
    expected_content: Any,
    should_raise: bool,
) -> None:
    upsert_response(
        kind="/parquet-and-dataset-info", dataset=dataset, content=upstream_content, http_status=upstream_status
    )
    job_runner = get_job_runner(dataset=dataset, app_config=app_config)
    if should_raise:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.type.__name__ == expected_error_code
    else:
        assert job_runner.compute() == expected_content


def test_doesnotexist(app_config: AppConfig, get_job_runner) -> None:
    dataset = "doesnotexist"
    job_runner = get_job_runner(dataset=dataset, app_config=app_config)
    with pytest.raises(DatasetNotFoundError):
        job_runner.compute()
