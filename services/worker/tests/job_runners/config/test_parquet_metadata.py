# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import io
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fsspec.implementations.http import HTTPFileSystem
from libcommon.exceptions import PreviousStepFormatError
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import CachedArtifactError, upsert_response
from libcommon.storage import StrPath
from libcommon.utils import Priority

from worker.config import AppConfig
from worker.job_runners.config.parquet import ConfigParquetResponse
from worker.job_runners.config.parquet_and_info import ParquetFileItem
from worker.job_runners.config.parquet_metadata import (
    ConfigParquetMetadataJobRunner,
    ConfigParquetMetadataResponse,
    ParquetFileMetadataItem,
)


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, str, AppConfig], ConfigParquetMetadataJobRunner]

dummy_parquet_buffer = io.BytesIO()
pq.write_table(pa.table({"a": [0, 1, 2]}), dummy_parquet_buffer)


@pytest.fixture
def get_job_runner(
    parquet_metadata_directory: StrPath,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
    ) -> ConfigParquetMetadataJobRunner:
        processing_step_name = ConfigParquetMetadataJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            {
                "dataset-level": {"input_type": "dataset"},
                processing_step_name: {
                    "input_type": "dataset",
                    "job_runner_version": ConfigParquetMetadataJobRunner.get_job_runner_version(),
                    "triggered_by": "dataset-level",
                },
            }
        )
        return ConfigParquetMetadataJobRunner(
            job_info={
                "type": ConfigParquetMetadataJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
                    "config": config,
                    "split": None,
                    "partition": None,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
            },
            app_config=app_config,
            processing_step=processing_graph.get_processing_step(processing_step_name),
            parquet_metadata_directory=parquet_metadata_directory,
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,config,upstream_status,upstream_content,expected_error_code,expected_content,should_raise",
    [
        (
            "ok",
            "config_1",
            HTTPStatus.OK,
            ConfigParquetResponse(
                parquet_files=[
                    ParquetFileItem(
                        dataset="ok", config="config_1", split="train", url="url1", filename="filename1", size=0
                    ),
                    ParquetFileItem(
                        dataset="ok", config="config_1", split="train", url="url2", filename="filename2", size=0
                    ),
                ],
            ),
            None,
            ConfigParquetMetadataResponse(
                parquet_files_metadata=[
                    ParquetFileMetadataItem(
                        dataset="ok",
                        config="config_1",
                        split="train",
                        url="url1",
                        filename="filename1",
                        size=0,
                        num_rows=3,
                        parquet_metadata_subpath="ok/--/config_1/filename1",
                    ),
                    ParquetFileMetadataItem(
                        dataset="ok",
                        config="config_1",
                        split="train",
                        url="url2",
                        filename="filename2",
                        size=0,
                        num_rows=3,
                        parquet_metadata_subpath="ok/--/config_1/filename2",
                    ),
                ]
            ),
            False,
        ),
        (
            "status_error",
            "config_1",
            HTTPStatus.NOT_FOUND,
            {"error": "error"},
            CachedArtifactError.__name__,
            None,
            True,
        ),
        (
            "format_error",
            "config_1",
            HTTPStatus.OK,
            {"not_parquet_files": "wrong_format"},
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
    config: str,
    upstream_status: HTTPStatus,
    upstream_content: Any,
    expected_error_code: str,
    expected_content: Any,
    should_raise: bool,
) -> None:
    upsert_response(
        kind="config-parquet",
        dataset=dataset,
        config=config,
        content=upstream_content,
        http_status=upstream_status,
    )
    job_runner = get_job_runner(dataset, config, app_config)
    if should_raise:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.type.__name__ == expected_error_code
    else:
        with patch("worker.job_runners.config.parquet_metadata.get_parquet_file") as mock_ParquetFile:
            mock_ParquetFile.return_value = pq.ParquetFile(dummy_parquet_buffer)
            assert job_runner.compute().content == expected_content
            assert mock_ParquetFile.call_count == len(upstream_content["parquet_files"])
            for parquet_file_item in upstream_content["parquet_files"]:
                mock_ParquetFile.assert_any_call(
                    parquet_file_item["url"], fs=HTTPFileSystem(), hf_token=app_config.common.hf_token
                )
        for parquet_file_metadata_item in expected_content["parquet_files_metadata"]:
            assert (
                pq.read_metadata(
                    Path(job_runner.parquet_metadata_directory)
                    / parquet_file_metadata_item["parquet_metadata_subpath"]
                )
                == pq.ParquetFile(dummy_parquet_buffer).metadata
            )
