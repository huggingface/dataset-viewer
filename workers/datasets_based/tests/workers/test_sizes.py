# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any

import pytest
from libcommon.simple_cache import _clean_cache_database, upsert_response

from datasets_based.config import AppConfig
from datasets_based.workers.sizes import (
    DatasetNotFoundError,
    PreviousStepFormatError,
    PreviousStepStatusError,
    SizesWorker,
)


@pytest.fixture(autouse=True)
def clean_mongo_database(app_config: AppConfig) -> None:
    _clean_cache_database()


def get_worker(dataset: str, app_config: AppConfig, force: bool = False) -> SizesWorker:
    return SizesWorker(
        job_info={
            "type": SizesWorker.get_job_type(),
            "dataset": dataset,
            "config": None,
            "split": None,
            "job_id": "job_id",
            "force": force,
        },
        app_config=app_config,
    )


@pytest.mark.parametrize(
    "dataset,upstream_status,upstream_content,expected_error_code,expected_content,should_raise",
    [
        (
            "dataset_ok",
            HTTPStatus.OK,
            {
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
                }
            },
            None,
            {
                "sizes": {
                    "dataset": {
                        "dataset": "dataset_ok",
                        "original_size_in_bytes": 21507144,
                        "parquet_size_in_bytes": 20394144,
                        "num_rows": 74000,
                    },
                    "configs": [
                        {
                            "dataset": "dataset_ok",
                            "config": "config_1",
                            "original_size_in_bytes": 11594722,
                            "parquet_size_in_bytes": 20387232,
                            "num_rows": 70000,
                            "num_columns": 2,
                        },
                        {
                            "dataset": "dataset_ok",
                            "config": "config_2",
                            "original_size_in_bytes": 9912422,
                            "parquet_size_in_bytes": 6912,
                            "num_rows": 4000,
                            "num_columns": 3,
                        },
                    ],
                    "splits": [
                        {
                            "dataset": "dataset_ok",
                            "config": "config_1",
                            "split": "train",
                            "parquet_size_in_bytes": 17470800,
                            "num_rows": 60000,
                            "num_columns": 2,
                        },
                        {
                            "dataset": "dataset_ok",
                            "config": "config_1",
                            "split": "test",
                            "parquet_size_in_bytes": 2916432,
                            "num_rows": 10000,
                            "num_columns": 2,
                        },
                        {
                            "dataset": "dataset_ok",
                            "config": "config_2",
                            "split": "train",
                            "parquet_size_in_bytes": 5678,
                            "num_rows": 3000,
                            "num_columns": 3,
                        },
                        {
                            "dataset": "dataset_ok",
                            "config": "config_2",
                            "split": "test",
                            "parquet_size_in_bytes": 1234,
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
    dataset: str,
    upstream_status: HTTPStatus,
    upstream_content: Any,
    expected_error_code: str,
    expected_content: Any,
    should_raise: bool,
) -> None:
    upsert_response(kind="/dataset-info", dataset=dataset, content=upstream_content, http_status=upstream_status)
    worker = get_worker(dataset=dataset, app_config=app_config)
    if should_raise:
        with pytest.raises(Exception) as e:
            worker.compute()
        assert e.type.__name__ == expected_error_code
    else:
        assert worker.compute() == expected_content


def test_doesnotexist(app_config: AppConfig) -> None:
    dataset = "doesnotexist"
    worker = get_worker(dataset=dataset, app_config=app_config)
    with pytest.raises(DatasetNotFoundError):
        worker.compute()
