# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import replace
from http import HTTPStatus
from typing import Callable

import pytest
from datasets.packaged_modules import csv
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response
from libcommon.storage import StrPath
from libcommon.utils import Priority

from worker.config import AppConfig
from worker.job_runners.split.first_rows_from_streaming import (
    SplitFirstRowsFromStreamingJobRunner,
)
from worker.resources import LibrariesResource
from worker.utils import get_json_size

from ...fixtures.hub import HubDatasets, get_default_config_split

GetJobRunner = Callable[[str, str, str, AppConfig], SplitFirstRowsFromStreamingJobRunner]


@pytest.fixture
def get_job_runner(
    assets_directory: StrPath,
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        split: str,
        app_config: AppConfig,
    ) -> SplitFirstRowsFromStreamingJobRunner:
        processing_step_name = SplitFirstRowsFromStreamingJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            {
                "dataset-level": {"input_type": "dataset"},
                "config-level": {"input_type": "dataset", "triggered_by": "dataset-level"},
                processing_step_name: {
                    "input_type": "dataset",
                    "job_runner_version": SplitFirstRowsFromStreamingJobRunner.get_job_runner_version(),
                    "triggered_by": "config-level",
                },
            }
        )
        return SplitFirstRowsFromStreamingJobRunner(
            job_info={
                "type": SplitFirstRowsFromStreamingJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
                    "config": config,
                    "split": split,
                    "partition_start": None,
                    "partition_end": None,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
            },
            app_config=app_config,
            processing_step=processing_graph.get_processing_step(processing_step_name),
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
            assets_directory=assets_directory,
        )

    return _get_job_runner


def test_compute(app_config: AppConfig, get_job_runner: GetJobRunner, hub_public_csv: str) -> None:
    dataset, config, split = get_default_config_split(hub_public_csv)
    job_runner = get_job_runner(dataset, config, split, app_config)
    upsert_response(
        kind="config-split-names-from-streaming",
        dataset=dataset,
        config=config,
        content={"splits": [{"dataset": dataset, "config": config, "split": split}]},
        http_status=HTTPStatus.OK,
    )
    response = job_runner.compute()
    assert response
    content = response.content
    assert content
    assert content["features"][0]["feature_idx"] == 0
    assert content["features"][0]["name"] == "col_1"
    assert content["features"][0]["type"]["_type"] == "Value"
    assert content["features"][0]["type"]["dtype"] == "int64"  # <---|
    assert content["features"][1]["type"]["dtype"] == "int64"  # <---|- auto-detected by the datasets library
    assert content["features"][2]["type"]["dtype"] == "float64"  # <-|


@pytest.mark.parametrize(
    "name,use_token,exception_name,cause",
    [
        ("public", False, None, None),
        ("audio", False, None, None),
        ("image", False, None, None),
        ("images_list", False, None, None),
        ("jsonl", False, None, None),
        ("gated", True, None, None),
        ("private", True, None, None),
        ("does_not_exist_config", False, "CachedArtifactError", None),
        # should we really test the following cases?
        # The assumption is that the dataset exists and is accessible with the token
        ("does_not_exist_split", False, "SplitNotFoundError", None),
        ("gated", False, "InfoError", "FileNotFoundError"),
        ("private", False, "InfoError", "FileNotFoundError"),
    ],
)
def test_number_rows(
    hub_datasets: HubDatasets,
    get_job_runner: GetJobRunner,
    name: str,
    use_token: bool,
    exception_name: str,
    cause: str,
    app_config: AppConfig,
) -> None:
    # temporary patch to remove the effect of
    # https://github.com/huggingface/datasets/issues/4875#issuecomment-1280744233
    # note: it fixes the tests, but it does not fix the bug in the "real world"
    if hasattr(csv, "_patched_for_streaming") and csv._patched_for_streaming:
        csv._patched_for_streaming = False

    dataset = hub_datasets[name]["name"]
    expected_first_rows_response = hub_datasets[name]["first_rows_response"]
    dataset, config, split = get_default_config_split(dataset)
    job_runner = get_job_runner(
        dataset,
        config,
        split,
        app_config if use_token else replace(app_config, common=replace(app_config.common, hf_token=None)),
    )

    if exception_name is None:
        upsert_response(
            kind="config-split-names-from-streaming",
            dataset=dataset,
            config=config,
            content={"splits": [{"dataset": dataset, "config": config, "split": split}]},
            http_status=HTTPStatus.OK,
        )
        result = job_runner.compute().content
        assert result == expected_first_rows_response
        return
    elif exception_name == "SplitNotFoundError":
        upsert_response(
            kind="config-split-names-from-streaming",
            dataset=dataset,
            config=config,
            content={"splits": [{"dataset": dataset, "config": config, "split": "other_split"}]},
            http_status=HTTPStatus.OK,
        )
    elif exception_name in {"InfoError", "SplitsNamesError"}:
        upsert_response(
            kind="config-split-names-from-streaming",
            dataset=dataset,
            config=config,
            content={"splits": [{"dataset": dataset, "config": config, "split": split}]},
            http_status=HTTPStatus.OK,
        )

    with pytest.raises(Exception) as exc_info:
        job_runner.compute()
    assert exc_info.typename == exception_name


@pytest.mark.parametrize(
    "name,rows_max_bytes,columns_max_number,error_code",
    [
        # not-truncated public response is 687 bytes
        ("public", 10, 1_000, "TooBigContentError"),  # too small limit, even with truncation
        ("public", 1_000, 1_000, None),  # not truncated
        ("public", 1_000, 1, "TooManyColumnsError"),  # too small columns limit
        # not-truncated big response is 5_885_989 bytes
        ("big", 10, 1_000, "TooBigContentError"),  # too small limit, even with truncation
        ("big", 1_000, 1_000, None),  # truncated successfully
        ("big", 10_000_000, 1_000, None),  # not truncated
    ],
)
def test_truncation(
    hub_datasets: HubDatasets,
    get_job_runner: GetJobRunner,
    app_config: AppConfig,
    name: str,
    rows_max_bytes: int,
    columns_max_number: int,
    error_code: str,
) -> None:
    dataset, config, split = get_default_config_split(hub_datasets[name]["name"])
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

    upsert_response(
        kind="config-split-names-from-streaming",
        dataset=dataset,
        config=config,
        content={"splits": [{"dataset": dataset, "config": config, "split": split}]},
        http_status=HTTPStatus.OK,
    )

    if error_code:
        with pytest.raises(CustomError) as error_info:
            job_runner.compute()
        assert error_info.value.code == error_code
    else:
        response = job_runner.compute().content
        assert get_json_size(response) <= rows_max_bytes
