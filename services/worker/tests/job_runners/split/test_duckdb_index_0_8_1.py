# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import replace
from http import HTTPStatus
from typing import Optional
from unittest.mock import patch

import datasets.config
import pytest
from datasets import Features
from datasets.packaged_modules.csv.csv import CsvConfig
from libcommon.dtos import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response
from libcommon.storage import StrPath

from worker.config import AppConfig
from worker.job_runners.config.parquet import ConfigParquetJobRunner
from worker.job_runners.config.parquet_and_info import ConfigParquetAndInfoJobRunner
from worker.job_runners.config.parquet_metadata import ConfigParquetMetadataJobRunner
from worker.job_runners.split.duckdb_index_0_8_1 import (
    SplitDuckDbIndex081JobRunner,
)
from worker.resources import LibrariesResource

from ...fixtures.hub import HubDatasetTest
from ..utils import REVISION_NAME

# TODO: Remove this file when all split-duckdb-index-010 entries have been computed

GetJobRunner = Callable[[str, str, str, AppConfig], SplitDuckDbIndex081JobRunner]

GetParquetAndInfoJobRunner = Callable[[str, str, AppConfig], ConfigParquetAndInfoJobRunner]
GetParquetJobRunner = Callable[[str, str, AppConfig], ConfigParquetJobRunner]
GetParquetMetadataJobRunner = Callable[[str, str, AppConfig], ConfigParquetMetadataJobRunner]


@pytest.fixture
def get_job_runner(
    parquet_metadata_directory: StrPath,
    duckdb_index_cache_directory: StrPath,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        split: str,
        app_config: AppConfig,
    ) -> SplitDuckDbIndex081JobRunner:
        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        upsert_response(
            kind="config-split-names-from-streaming",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            config=config,
            content={"splits": [{"dataset": dataset, "config": config, "split": split}]},
            http_status=HTTPStatus.OK,
        )

        return SplitDuckDbIndex081JobRunner(
            job_info={
                "type": SplitDuckDbIndex081JobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": REVISION_NAME,
                    "config": config,
                    "split": split,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 50,
            },
            app_config=app_config,
            duckdb_index_cache_directory=duckdb_index_cache_directory,
            parquet_metadata_directory=parquet_metadata_directory,
        )

    return _get_job_runner


@pytest.fixture
def get_parquet_and_info_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetParquetAndInfoJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
    ) -> ConfigParquetAndInfoJobRunner:
        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        return ConfigParquetAndInfoJobRunner(
            job_info={
                "type": ConfigParquetAndInfoJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": REVISION_NAME,
                    "config": config,
                    "split": None,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 50,
            },
            app_config=app_config,
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
        )

    return _get_job_runner


@pytest.fixture
def get_parquet_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetParquetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
    ) -> ConfigParquetJobRunner:
        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        return ConfigParquetJobRunner(
            job_info={
                "type": ConfigParquetJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
                    "config": config,
                    "split": None,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 50,
            },
            app_config=app_config,
        )

    return _get_job_runner


@pytest.fixture
def get_parquet_metadata_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
    parquet_metadata_directory: StrPath,
) -> GetParquetMetadataJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
    ) -> ConfigParquetMetadataJobRunner:
        upsert_response(
            kind="dataset-config-names",
            dataset_git_revision=REVISION_NAME,
            dataset=dataset,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        return ConfigParquetMetadataJobRunner(
            job_info={
                "type": ConfigParquetMetadataJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
                    "config": config,
                    "split": None,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 50,
            },
            app_config=app_config,
            parquet_metadata_directory=parquet_metadata_directory,
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "hub_dataset_name,max_split_size_bytes,expected_has_fts,expected_partial,expected_error_code",
    [
        ("duckdb_index", None, True, False, None),
        ("duckdb_index_from_partial_export", None, True, True, None),
        ("gated", None, True, False, None),
        ("partial_duckdb_index_from_multiple_files_public", 1, False, True, None),
    ],
)
def test_compute(
    get_parquet_and_info_job_runner: GetParquetAndInfoJobRunner,
    get_parquet_job_runner: GetParquetJobRunner,
    get_parquet_metadata_job_runner: GetParquetMetadataJobRunner,
    get_job_runner: GetJobRunner,
    app_config: AppConfig,
    hub_responses_public: HubDatasetTest,
    hub_responses_duckdb_index: HubDatasetTest,
    hub_responses_gated_duckdb_index: HubDatasetTest,
    hub_dataset_name: str,
    max_split_size_bytes: Optional[int],
    expected_has_fts: bool,
    expected_partial: bool,
    expected_error_code: str,
) -> None:
    hub_datasets = {
        "duckdb_index": hub_responses_duckdb_index,
        "duckdb_index_from_partial_export": hub_responses_duckdb_index,
        "gated": hub_responses_gated_duckdb_index,
        "partial_duckdb_index_from_multiple_files_public": hub_responses_public,
    }
    dataset = hub_datasets[hub_dataset_name]["name"]
    config = hub_datasets[hub_dataset_name]["config_names_response"]["config_names"][0]["config"]
    split = "train"
    partial_parquet_export = hub_dataset_name == "duckdb_index_from_partial_export"
    multiple_parquet_files = hub_dataset_name == "partial_duckdb_index_from_multiple_files_public"

    app_config = (
        app_config
        if max_split_size_bytes is None
        else replace(
            app_config, duckdb_index=replace(app_config.duckdb_index, max_split_size_bytes=max_split_size_bytes)
        )
    )
    app_config = (
        app_config
        if not partial_parquet_export
        else replace(
            app_config,
            parquet_and_info=replace(
                app_config.parquet_and_info, max_dataset_size_bytes=1, max_row_group_byte_size_for_copy=1
            ),
        )
    )

    parquet_and_info_job_runner = get_parquet_and_info_job_runner(dataset, config, app_config)
    with ExitStack() as stack:
        if multiple_parquet_files:
            stack.enter_context(patch.object(datasets.config, "MAX_SHARD_SIZE", 1))
            # Set a small chunk size to yield more than one Arrow Table in _generate_tables
            # to be able to generate multiple tables and therefore multiple files
            stack.enter_context(patch.object(CsvConfig, "pd_read_csv_kwargs", {"chunksize": 1}))
        parquet_and_info_response = parquet_and_info_job_runner.compute()
    config_parquet_and_info = parquet_and_info_response.content
    if multiple_parquet_files:
        assert len(config_parquet_and_info["parquet_files"]) > 1

    assert config_parquet_and_info["partial"] is partial_parquet_export

    upsert_response(
        "config-parquet-and-info",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        http_status=HTTPStatus.OK,
        content=config_parquet_and_info,
    )

    parquet_job_runner = get_parquet_job_runner(dataset, config, app_config)
    parquet_response = parquet_job_runner.compute()
    config_parquet = parquet_response.content

    assert config_parquet["partial"] is partial_parquet_export

    upsert_response(
        "config-parquet",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        http_status=HTTPStatus.OK,
        content=config_parquet,
    )

    parquet_metadata_job_runner = get_parquet_metadata_job_runner(dataset, config, app_config)
    parquet_metadata_response = parquet_metadata_job_runner.compute()
    config_parquet_metadata = parquet_metadata_response.content
    parquet_export_num_rows = sum(
        parquet_file["num_rows"] for parquet_file in config_parquet_metadata["parquet_files_metadata"]
    )

    assert config_parquet_metadata["partial"] is partial_parquet_export

    upsert_response(
        "config-parquet-metadata",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        http_status=HTTPStatus.OK,
        content=config_parquet_metadata,
    )

    # setup is ready, test starts here
    job_runner = get_job_runner(dataset, config, split, app_config)
    job_runner.pre_compute()

    if expected_error_code:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.typename == expected_error_code
    else:
        response = job_runner.compute()
        assert response
        content = response.content
        url = content["url"]
        file_name = content["filename"]
        features = content["features"]
        has_fts = content["has_fts"]
        partial = content["partial"]
        assert isinstance(has_fts, bool)
        assert has_fts == expected_has_fts
        assert isinstance(url, str)
        if partial_parquet_export:
            assert url.rsplit("/", 2)[1] == "partial-" + split
        else:
            assert url.rsplit("/", 2)[1] == split
        assert file_name is not None
        assert Features.from_dict(features) is not None
        assert isinstance(partial, bool)
        assert partial == expected_partial
        if content["num_rows"] < parquet_export_num_rows:
            assert url.rsplit("/", 1)[1] == "partial-index.duckdb"
        else:
            assert url.rsplit("/", 1)[1] == "index.duckdb"

    job_runner.post_compute()
