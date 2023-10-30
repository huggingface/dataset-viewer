# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Optional

import pytest
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource
from libcommon.s3_client import S3Client
from libcommon.simple_cache import upsert_response
from libcommon.storage import StrPath
from libcommon.utils import JobInfo, Priority

from worker.config import AppConfig
from worker.job_runner_factory import JobRunnerFactory
from worker.resources import LibrariesResource

from .job_runners.utils import REVISION_NAME


@pytest.fixture(autouse=True)
def cache_mongo_resource_autouse(cache_mongo_resource: CacheMongoResource) -> CacheMongoResource:
    return cache_mongo_resource


@pytest.fixture()
def processing_graph(app_config: AppConfig) -> ProcessingGraph:
    return ProcessingGraph(app_config.processing_graph)


@pytest.mark.parametrize(
    "level,job_type,expected_job_runner",
    [
        ("dataset", "dataset-config-names", "DatasetConfigNamesJobRunner"),
        ("split", "split-first-rows-from-streaming", "SplitFirstRowsFromStreamingJobRunner"),
        ("config", "config-parquet-and-info", "ConfigParquetAndInfoJobRunner"),
        ("config", "config-parquet", "ConfigParquetJobRunner"),
        ("dataset", "dataset-parquet", "DatasetParquetJobRunner"),
        ("config", "config-info", "ConfigInfoJobRunner"),
        ("dataset", "dataset-info", "DatasetInfoJobRunner"),
        ("config", "config-size", "ConfigSizeJobRunner"),
        ("dataset", "dataset-size", "DatasetSizeJobRunner"),
        (None, "/unknown", None),
    ],
)
def test_create_job_runner(
    app_config: AppConfig,
    processing_graph: ProcessingGraph,
    libraries_resource: LibrariesResource,
    assets_directory: StrPath,
    parquet_metadata_directory: StrPath,
    duckdb_index_cache_directory: StrPath,
    statistics_cache_directory: StrPath,
    level: Optional[str],
    job_type: str,
    expected_job_runner: Optional[str],
) -> None:
    s3_client = S3Client(
        region_name="us-east-1",
        aws_access_key_id="access_key_id",
        aws_secret_access_key="secret_access_key",
        bucket_name="bucket",
    )
    factory = JobRunnerFactory(
        app_config=app_config,
        processing_graph=processing_graph,
        hf_datasets_cache=libraries_resource.hf_datasets_cache,
        assets_directory=assets_directory,
        parquet_metadata_directory=parquet_metadata_directory,
        duckdb_index_cache_directory=duckdb_index_cache_directory,
        statistics_cache_directory=statistics_cache_directory,
        s3_client=s3_client,
    )
    dataset, config, split = "dataset", "config", "split"
    job_info: JobInfo = {
        "type": job_type,
        "params": {
            "dataset": dataset,
            "revision": REVISION_NAME,
            "config": config,
            "split": split,
        },
        "job_id": "job_id",
        "priority": Priority.NORMAL,
        "difficulty": 50,
    }

    if level in {"split", "config"}:
        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

    if level == "split":
        upsert_response(
            kind="config-split-names-from-streaming",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            config=config,
            content={"splits": [{"dataset": dataset, "config": config, "split": split}]},
            http_status=HTTPStatus.OK,
        )

    if expected_job_runner is None:
        with pytest.raises(KeyError):
            factory.create_job_runner(job_info=job_info)
    else:
        job_runner = factory.create_job_runner(job_info=job_info)
        assert job_runner.__class__.__name__ == expected_job_runner
