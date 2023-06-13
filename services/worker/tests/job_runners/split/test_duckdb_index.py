# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from http import HTTPStatus
from typing import Callable

import pytest
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response
from libcommon.storage import StrPath
from libcommon.utils import Priority

from worker.config import AppConfig
from worker.job_runners.config.parquet_and_info import ConfigParquetAndInfoJobRunner
from worker.job_runners.split.duckdb_index import SplitDuckDbIndexJobRunner
from worker.resources import LibrariesResource

from ...fixtures.hub import HubDatasets

GetJobRunner = Callable[[str, str, str, AppConfig], SplitDuckDbIndexJobRunner]

GetParquetJobRunner = Callable[[str, str, AppConfig], ConfigParquetAndInfoJobRunner]


@pytest.fixture
def get_job_runner(
    duckdb_index_directory: StrPath,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        split: str,
        app_config: AppConfig,
    ) -> SplitDuckDbIndexJobRunner:
        processing_step_name = SplitDuckDbIndexJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            {
                "dataset-step": {"input_type": "dataset"},
                "config-parquet": {
                    "input_type": "config",
                    "triggered_by": "dataset-step",
                    "provides_config_parquet": True,
                },
                "config-split-names-from-streaming": {
                    "input_type": "config",
                    "triggered_by": "dataset-step",
                },
                processing_step_name: {
                    "input_type": "dataset",
                    "job_runner_version": SplitDuckDbIndexJobRunner.get_job_runner_version(),
                    "triggered_by": ["config-parquet", "config-split-names-from-streaming"],
                },
            }
        )
        return SplitDuckDbIndexJobRunner(
            job_info={
                "type": SplitDuckDbIndexJobRunner.get_job_type(),
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
            duckdb_index_directory=duckdb_index_directory,
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
    ) -> ConfigParquetAndInfoJobRunner:
        processing_step_name = ConfigParquetAndInfoJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            {
                "dataset-level": {"input_type": "dataset"},
                processing_step_name: {
                    "input_type": "config",
                    "job_runner_version": ConfigParquetAndInfoJobRunner.get_job_runner_version(),
                    "triggered_by": "dataset-level",
                },
            }
        )
        return ConfigParquetAndInfoJobRunner(
            job_info={
                "type": ConfigParquetAndInfoJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
                    "config": config,
                    "split": None,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
            },
            app_config=app_config,
            processing_step=processing_graph.get_processing_step(processing_step_name),
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "hub_dataset_name,expected_error_code",
    [
        ("duckdb_index", None),
        ("text_image", "UnsupportedIndexableColumnsError"),
        ("public", "NoIndexableColumnsError"),
    ],
)
def test_compute(
    get_parquet_job_runner: GetParquetJobRunner,
    get_job_runner: GetJobRunner,
    app_config: AppConfig,
    hub_datasets: HubDatasets,
    hub_dataset_name: str,
    expected_error_code: str,
) -> None:
    dataset = hub_datasets[hub_dataset_name]["name"]
    config_names = hub_datasets[hub_dataset_name]["config_names_response"]
    config = hub_datasets[hub_dataset_name]["config_names_response"]["config_names"][0]["config"]
    splits_response = hub_datasets[hub_dataset_name]["splits_response"]
    split = "train"

    upsert_response(
        "dataset-config-names",
        dataset=dataset,
        http_status=HTTPStatus.OK,
        content=config_names,
    )

    upsert_response(
        "config-split-names-from-streaming",
        dataset=dataset,
        config=config,
        http_status=HTTPStatus.OK,
        content=splits_response,
    )

    parquet_job_runner = get_parquet_job_runner(dataset, config, app_config)
    parquet_response = parquet_job_runner.compute()
    config_parquet = parquet_response.content

    upsert_response(
        "config-parquet",
        dataset=dataset,
        config=config,
        http_status=HTTPStatus.OK,
        content=config_parquet,
    )

    assert parquet_response
    job_runner = get_job_runner(dataset, config, split, app_config)

    if expected_error_code:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.typename == expected_error_code
    else:
        job_runner.pre_compute()
        response = job_runner.compute()
        assert response
        content = response.content
        assert content["url"] is not None
        assert content["filename"] is not None
        job_runner.post_compute()
