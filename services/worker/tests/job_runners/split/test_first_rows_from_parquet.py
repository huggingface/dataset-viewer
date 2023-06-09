# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import replace
from http import HTTPStatus
from typing import Callable, Generator, List
from unittest.mock import patch

import pytest
from datasets import Dataset
from fsspec import AbstractFileSystem
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response
from libcommon.storage import StrPath
from libcommon.utils import Priority

from worker.config import AppConfig
from worker.job_runners.split.first_rows_from_parquet import (
    SplitFirstRowsFromParquetJobRunner,
)
from worker.utils import get_json_size

GetJobRunner = Callable[[str, str, str, AppConfig], SplitFirstRowsFromParquetJobRunner]


@pytest.fixture
def get_job_runner(
    assets_directory: StrPath,
    parquet_metadata_directory: StrPath,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        split: str,
        app_config: AppConfig,
    ) -> SplitFirstRowsFromParquetJobRunner:
        processing_step_name = SplitFirstRowsFromParquetJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            {
                "dataset-level": {"input_type": "dataset"},
                "config-level": {
                    "input_type": "config",
                    "triggered_by": "dataset-level",
                    "provides_config_parquet": True,
                },
                processing_step_name: {
                    "input_type": "dataset",
                    "job_runner_version": SplitFirstRowsFromParquetJobRunner.get_job_runner_version(),
                    "triggered_by": "config-level",
                },
            }
        )
        return SplitFirstRowsFromParquetJobRunner(
            job_info={
                "type": SplitFirstRowsFromParquetJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
                    "config": config,
                    "split": split,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
            },
            app_config=app_config,
            processing_step=processing_graph.get_processing_step(processing_step_name),
            processing_graph=processing_graph,
            assets_directory=assets_directory,
            parquet_metadata_directory=parquet_metadata_directory,
        )

    return _get_job_runner


@pytest.fixture
def ds() -> Dataset:
    return Dataset.from_dict({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})


@pytest.fixture
def ds_fs(ds: Dataset, tmpfs: AbstractFileSystem) -> Generator[AbstractFileSystem, None, None]:
    with tmpfs.open("config/dataset-split.parquet", "wb") as f:
        ds.to_parquet(f)
    yield tmpfs


def mock_get_hf_parquet_uris(paths: List[str], dataset: str) -> List[str]:
    return paths


@pytest.mark.parametrize(
    "rows_max_bytes,columns_max_number,error_code",
    [
        (0, 10, "TooBigContentError"),  # too small limit, even with truncation
        (1_000, 1, "TooManyColumnsError"),  # too small columns limit
        (1_000, 10, None),
    ],
)
def test_compute(
    ds_fs: AbstractFileSystem,
    get_job_runner: GetJobRunner,
    app_config: AppConfig,
    rows_max_bytes: int,
    columns_max_number: int,
    error_code: str,
) -> None:
    dataset, config, split = "dataset", "config", "split"

    config_parquet_content = {
        "parquet_files": [
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "url": "https://fake.huggingface.co/datasets/dataset/resolve/refs%2Fconvert%2Fparquet/config/dataset-split.parquet",  # noqa: E501
                "filename": "dataset-split.parquet",
                "size": 128,
            }
        ]
    }

    upsert_response(
        kind="config-level",
        dataset=dataset,
        config=config,
        content=config_parquet_content,
        http_status=HTTPStatus.OK,
    )

    with patch("libcommon.parquet_utils.get_hf_fs", return_value=ds_fs):
        with patch(
            "libcommon.parquet_utils.get_hf_parquet_uris",
            side_effect=mock_get_hf_parquet_uris,
        ):
            job_runner = get_job_runner(
                dataset,
                config,
                split,
                replace(
                    app_config,
                    common=replace(app_config.common, hf_token=None),
                    first_rows=replace(
                        app_config.first_rows,
                        max_number=1_000_000,
                        min_number=10,
                        max_bytes=rows_max_bytes,
                        min_cell_bytes=10,
                        columns_max_number=columns_max_number,
                    ),
                ),
            )

            if error_code:
                with pytest.raises(CustomError) as error_info:
                    job_runner.compute()
                assert error_info.value.code == error_code
            else:
                response = job_runner.compute().content
                assert get_json_size(response) <= rows_max_bytes
                assert response
                assert response["rows"]
                assert response["features"]
                assert len(response["rows"]) == 3  # testing file has 3 rows see config/dataset-split.parquet file
                assert (
                    len(response["features"]) == 2
                )  # testing file has 2 columns see config/dataset-split.parquet file
                assert response["features"][0]["feature_idx"] == 0
                assert response["features"][0]["name"] == "col1"
                assert response["features"][0]["type"]["_type"] == "Value"
                assert response["features"][0]["type"]["dtype"] == "int64"
                assert response["features"][1]["feature_idx"] == 1
                assert response["features"][1]["name"] == "col2"
                assert response["features"][1]["type"]["_type"] == "Value"
                assert response["features"][1]["type"]["dtype"] == "string"
                assert response["rows"][0]["row_idx"] == 0
                assert response["rows"][0]["truncated_cells"] == []
                assert response["rows"][0]["row"] == {"col1": 1, "col2": "a"}
                assert response["rows"][1]["row_idx"] == 1
                assert response["rows"][1]["truncated_cells"] == []
                assert response["rows"][1]["row"] == {"col1": 2, "col2": "b"}
                assert response["rows"][2]["row_idx"] == 2
                assert response["rows"][2]["truncated_cells"] == []
                assert response["rows"][2]["row"] == {"col1": 3, "col2": "c"}
