# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import io
from http import HTTPStatus
from typing import Any, Callable, Iterator, List, Optional

import datasets.builder
import pandas as pd
import pytest
import requests
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import DoesNotExist, get_response

from worker.config import AppConfig, ParquetAndInfoConfig
from worker.job_runners.parquet_and_dataset_info import (
    DatasetInBlockListError,
    DatasetTooBigFromDatasetsError,
    DatasetTooBigFromHubError,
    DatasetWithTooBigExternalFilesError,
    DatasetWithTooManyExternalFilesError,
    ParquetAndDatasetInfoJobRunner,
    get_dataset_info_or_raise,
    parse_repo_filename,
    raise_if_blocked,
    raise_if_not_supported,
    raise_if_too_big_from_datasets,
    raise_if_too_big_from_external_data_files,
    raise_if_too_big_from_hub,
)
from worker.resources import LibrariesResource

from ..fixtures.hub import HubDatasets


# see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
@pytest.fixture(scope="module", autouse=True)
def set_supported_datasets(hub_datasets: HubDatasets) -> Iterator[pytest.MonkeyPatch]:
    mp = pytest.MonkeyPatch()
    mp.setenv(
        "PARQUET_AND_INFO_BLOCKED_DATASETS",
        ",".join(value["name"] for value in hub_datasets.values() if "jsonl" in value["name"]),
    )
    mp.setenv(
        "PARQUET_AND_INFO_SUPPORTED_DATASETS",
        ",".join(value["name"] for value in hub_datasets.values() if "big" not in value["name"]),
    )
    yield mp
    mp.undo()


@pytest.fixture
def parquet_and_dataset_info_config(
    set_env_vars: pytest.MonkeyPatch, set_supported_datasets: pytest.MonkeyPatch
) -> ParquetAndInfoConfig:
    return ParquetAndInfoConfig.from_env()


GetJobRunner = Callable[[str, AppConfig, ParquetAndInfoConfig, bool], ParquetAndDatasetInfoJobRunner]


@pytest.fixture
def get_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
        parquet_and_dataset_info_config: ParquetAndInfoConfig,
        force: bool = False,
    ) -> ParquetAndDatasetInfoJobRunner:
        return ParquetAndDatasetInfoJobRunner(
            job_info={
                "type": ParquetAndDatasetInfoJobRunner.get_job_type(),
                "dataset": dataset,
                "config": None,
                "split": None,
                "job_id": "job_id",
                "force": force,
                "priority": Priority.NORMAL,
            },
            app_config=app_config,
            processing_step=ProcessingStep(
                name=ParquetAndDatasetInfoJobRunner.get_job_type(),
                input_type="dataset",
                requires=None,
                required_by_dataset_viewer=False,
                parent=None,
                ancestors=[],
                children=[],
                job_runner_version=ParquetAndDatasetInfoJobRunner.get_job_runner_version(),
            ),
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
            parquet_and_dataset_info_config=parquet_and_dataset_info_config,
        )

    return _get_job_runner


def assert_content_is_equal(content: Any, expected: Any) -> None:
    print(content)
    assert set(content.keys()) == {"parquet_files", "dataset_info"}, content
    assert content["parquet_files"] == expected["parquet_files"], content
    assert len(content["dataset_info"]) == 1, content
    content_value = list(content["dataset_info"].values())[0]
    expected_value = list(expected["dataset_info"].values())[0]
    assert set(content_value.keys()) == set(expected_value.keys()), content
    for key in content_value.keys():
        if key != "download_checksums":
            assert content_value[key] == expected_value[key], content
    assert len(content_value["download_checksums"]) == 1, content
    content_checksum = list(content_value["download_checksums"].values())[0]
    expected_checksum = list(expected_value["download_checksums"].values())[0]
    assert content_checksum == expected_checksum, content


def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    parquet_and_dataset_info_config: ParquetAndInfoConfig,
    hub_datasets: HubDatasets,
) -> None:
    dataset = hub_datasets["public"]["name"]
    job_runner = get_job_runner(dataset, app_config, parquet_and_dataset_info_config, False)
    assert job_runner.process()
    cached_response = get_response(kind=job_runner.processing_step.cache_kind, dataset=dataset)
    assert cached_response["http_status"] == HTTPStatus.OK
    assert cached_response["error_code"] is None
    assert cached_response["job_runner_version"] == job_runner.get_job_runner_version()
    assert cached_response["dataset_git_revision"] is not None
    content = cached_response["content"]
    assert len(content["parquet_files"]) == 1
    assert_content_is_equal(content, hub_datasets["public"]["parquet_and_dataset_info_response"])


def test_doesnotexist(
    app_config: AppConfig, get_job_runner: GetJobRunner, parquet_and_dataset_info_config: ParquetAndInfoConfig
) -> None:
    dataset = "doesnotexist"
    job_runner = get_job_runner(dataset, app_config, parquet_and_dataset_info_config, False)
    assert not job_runner.process()
    with pytest.raises(DoesNotExist):
        get_response(kind=job_runner.processing_step.cache_kind, dataset=dataset)


@pytest.mark.parametrize(
    "dataset,blocked,raises",
    [
        ("public", ["public"], True),
        ("public", ["public", "audio"], True),
        ("public", ["audio"], False),
        ("public", [], False),
    ],
)
def test_raise_if_blocked(dataset: str, blocked: List[str], raises: bool) -> None:
    if raises:
        with pytest.raises(DatasetInBlockListError):
            raise_if_blocked(dataset=dataset, blocked_datasets=blocked)
    else:
        raise_if_blocked(dataset=dataset, blocked_datasets=blocked)


@pytest.mark.parametrize(
    "name,raises",
    [("public", False), ("big", True)],
)
def test_raise_if_too_big_from_hub(
    hub_datasets: HubDatasets,
    name: str,
    raises: bool,
    app_config: AppConfig,
    parquet_and_dataset_info_config: ParquetAndInfoConfig,
) -> None:
    dataset = hub_datasets[name]["name"]
    dataset_info = get_dataset_info_or_raise(
        dataset=dataset,
        hf_endpoint=app_config.common.hf_endpoint,
        hf_token=app_config.common.hf_token,
        revision="main",
    )
    if raises:
        with pytest.raises(DatasetTooBigFromHubError):
            raise_if_too_big_from_hub(
                dataset_info=dataset_info, max_dataset_size=parquet_and_dataset_info_config.max_dataset_size
            )
    else:
        raise_if_too_big_from_hub(
            dataset_info=dataset_info, max_dataset_size=parquet_and_dataset_info_config.max_dataset_size
        )


@pytest.mark.parametrize(
    "name,raises",
    [("public", False), ("big", True)],
)
def test_raise_if_too_big_from_datasets(
    hub_datasets: HubDatasets,
    name: str,
    raises: bool,
    app_config: AppConfig,
    parquet_and_dataset_info_config: ParquetAndInfoConfig,
) -> None:
    dataset = hub_datasets[name]["name"]
    if raises:
        with pytest.raises(DatasetTooBigFromDatasetsError):
            raise_if_too_big_from_datasets(
                dataset=dataset,
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
                revision="main",
                max_dataset_size=parquet_and_dataset_info_config.max_dataset_size,
            )
    else:
        raise_if_too_big_from_datasets(
            dataset=dataset,
            hf_endpoint=app_config.common.hf_endpoint,
            hf_token=app_config.common.hf_token,
            revision="main",
            max_dataset_size=parquet_and_dataset_info_config.max_dataset_size,
        )


@pytest.mark.parametrize(
    "max_dataset_size,max_external_data_files,raises",
    [
        (None, None, False),
        (10, None, True),
    ],
)
def test_raise_if_too_big_external_files(
    external_files_dataset_builder: "datasets.builder.DatasetBuilder",
    raises: bool,
    max_dataset_size: Optional[int],
    max_external_data_files: Optional[int],
    app_config: AppConfig,
    parquet_and_dataset_info_config: ParquetAndInfoConfig,
) -> None:
    max_dataset_size = max_dataset_size or parquet_and_dataset_info_config.max_dataset_size
    max_external_data_files = max_external_data_files or parquet_and_dataset_info_config.max_external_data_files
    if raises:
        with pytest.raises(DatasetWithTooBigExternalFilesError):
            raise_if_too_big_from_external_data_files(
                builder=external_files_dataset_builder,
                hf_token=app_config.common.hf_token,
                max_dataset_size=max_dataset_size,
                max_external_data_files=max_external_data_files,
            )
    else:
        raise_if_too_big_from_external_data_files(
            builder=external_files_dataset_builder,
            hf_token=app_config.common.hf_token,
            max_dataset_size=max_dataset_size,
            max_external_data_files=max_external_data_files,
        )


@pytest.mark.parametrize(
    "max_dataset_size,max_external_data_files,raises",
    [
        (None, None, False),
        (None, 1, True),
    ],
)
def test_raise_if_too_many_external_files(
    external_files_dataset_builder: "datasets.builder.DatasetBuilder",
    raises: bool,
    max_dataset_size: Optional[int],
    max_external_data_files: Optional[int],
    app_config: AppConfig,
    parquet_and_dataset_info_config: ParquetAndInfoConfig,
) -> None:
    max_dataset_size = max_dataset_size or parquet_and_dataset_info_config.max_dataset_size
    max_external_data_files = max_external_data_files or parquet_and_dataset_info_config.max_external_data_files
    if raises:
        with pytest.raises(DatasetWithTooManyExternalFilesError):
            raise_if_too_big_from_external_data_files(
                builder=external_files_dataset_builder,
                hf_token=app_config.common.hf_token,
                max_dataset_size=max_dataset_size,
                max_external_data_files=max_external_data_files,
            )
    else:
        raise_if_too_big_from_external_data_files(
            builder=external_files_dataset_builder,
            hf_token=app_config.common.hf_token,
            max_dataset_size=max_dataset_size,
            max_external_data_files=max_external_data_files,
        )


@pytest.mark.parametrize(
    "in_list,raises",
    [
        (True, False),
        (False, True),
    ],
)
def test_raise_if_not_supported(
    hub_public_big: str,
    app_config: AppConfig,
    parquet_and_dataset_info_config: ParquetAndInfoConfig,
    in_list: bool,
    raises: bool,
) -> None:
    if raises:
        with pytest.raises(DatasetTooBigFromDatasetsError):
            raise_if_not_supported(
                dataset=hub_public_big,
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
                committer_hf_token=parquet_and_dataset_info_config.committer_hf_token,
                revision="main",
                max_dataset_size=parquet_and_dataset_info_config.max_dataset_size,
                supported_datasets=[hub_public_big] if in_list else ["another_dataset"],
                blocked_datasets=[],
            )
    else:
        raise_if_not_supported(
            dataset=hub_public_big,
            hf_endpoint=app_config.common.hf_endpoint,
            hf_token=app_config.common.hf_token,
            committer_hf_token=parquet_and_dataset_info_config.committer_hf_token,
            revision="main",
            max_dataset_size=parquet_and_dataset_info_config.max_dataset_size,
            supported_datasets=[hub_public_big] if in_list else ["another_dataset"],
            blocked_datasets=[],
        )


def test_not_supported_if_big(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    parquet_and_dataset_info_config: ParquetAndInfoConfig,
    hub_public_big: str,
) -> None:
    # Not in the list of supported datasets and bigger than the maximum size
    dataset = hub_public_big
    job_runner = get_job_runner(dataset, app_config, parquet_and_dataset_info_config, False)
    assert not job_runner.process()
    cached_response = get_response(kind=job_runner.processing_step.cache_kind, dataset=dataset)
    assert cached_response["http_status"] == HTTPStatus.NOT_IMPLEMENTED
    assert cached_response["error_code"] == "DatasetTooBigFromDatasetsError"


def test_supported_if_gated(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    parquet_and_dataset_info_config: ParquetAndInfoConfig,
    hub_gated_csv: str,
) -> None:
    # Access must be granted
    dataset = hub_gated_csv
    job_runner = get_job_runner(dataset, app_config, parquet_and_dataset_info_config, False)
    assert job_runner.process()
    cached_response = get_response(kind=job_runner.processing_step.cache_kind, dataset=dataset)
    assert cached_response["http_status"] == HTTPStatus.OK
    assert cached_response["error_code"] is None


def test_not_supported_if_gated_with_extra_fields(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    parquet_and_dataset_info_config: ParquetAndInfoConfig,
    hub_gated_extra_fields_csv: str,
) -> None:
    # Access request should fail because extra fields in gated datasets are not supported
    dataset = hub_gated_extra_fields_csv
    job_runner = get_job_runner(dataset, app_config, parquet_and_dataset_info_config, False)
    assert not job_runner.process()
    cached_response = get_response(kind=job_runner.processing_step.cache_kind, dataset=dataset)
    assert cached_response["http_status"] == HTTPStatus.NOT_FOUND
    assert cached_response["error_code"] == "GatedExtraFieldsError"


def test_blocked(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    parquet_and_dataset_info_config: ParquetAndInfoConfig,
    hub_public_jsonl: str,
) -> None:
    # In the list of blocked datasets
    dataset = hub_public_jsonl
    job_runner = get_job_runner(dataset, app_config, parquet_and_dataset_info_config, False)
    assert not job_runner.process()
    cached_response = get_response(kind=job_runner.processing_step.cache_kind, dataset=dataset)
    assert cached_response["http_status"] == HTTPStatus.NOT_IMPLEMENTED
    assert cached_response["error_code"] == "DatasetInBlockListError"


@pytest.mark.parametrize(
    "name",
    ["public", "audio", "gated"],
)
def test_compute_splits_response_simple_csv_ok(
    hub_datasets: HubDatasets,
    get_job_runner: GetJobRunner,
    name: str,
    app_config: AppConfig,
    parquet_and_dataset_info_config: ParquetAndInfoConfig,
    data_df: pd.DataFrame,
) -> None:
    dataset = hub_datasets[name]["name"]
    expected_parquet_and_dataset_info_response = hub_datasets[name]["parquet_and_dataset_info_response"]
    job_runner = get_job_runner(dataset, app_config, parquet_and_dataset_info_config, False)
    result = job_runner.compute().content
    assert_content_is_equal(result, expected_parquet_and_dataset_info_response)

    # download the parquet file and check that it is valid
    if name == "audio":
        return

    if name == "public":
        df = pd.read_parquet(result["parquet_files"][0]["url"], engine="auto")
    else:
        # in all these cases, the parquet files are not accessible without a token
        with pytest.raises(Exception):
            pd.read_parquet(result["parquet_files"][0]["url"], engine="auto")
        r = requests.get(
            result["parquet_files"][0]["url"], headers={"Authorization": f"Bearer {app_config.common.hf_token}"}
        )
        assert r.status_code == HTTPStatus.OK, r.text
        df = pd.read_parquet(io.BytesIO(r.content), engine="auto")
    assert df.equals(data_df), df


@pytest.mark.parametrize(
    "name,error_code,cause",
    [
        ("empty", "EmptyDatasetError", "EmptyDatasetError"),
        ("does_not_exist", "DatasetNotFoundError", "HTTPError"),
        ("gated_extra_fields", "GatedExtraFieldsError", "HTTPError"),
        ("private", "DatasetNotFoundError", None),
    ],
)
def test_compute_splits_response_simple_csv_error(
    hub_datasets: HubDatasets,
    get_job_runner: GetJobRunner,
    name: str,
    error_code: str,
    cause: str,
    app_config: AppConfig,
    parquet_and_dataset_info_config: ParquetAndInfoConfig,
) -> None:
    dataset = hub_datasets[name]["name"]
    job_runner = get_job_runner(dataset, app_config, parquet_and_dataset_info_config, False)
    with pytest.raises(CustomError) as exc_info:
        job_runner.compute()
    assert exc_info.value.code == error_code
    assert exc_info.value.cause_exception == cause
    if exc_info.value.disclose_cause:
        response = exc_info.value.as_response()
        assert set(response.keys()) == {"error", "cause_exception", "cause_message", "cause_traceback"}
        response_dict = dict(response)
        # ^ to remove mypy warnings
        assert response_dict["cause_exception"] == cause
        assert isinstance(response_dict["cause_traceback"], list)
        assert response_dict["cause_traceback"][0] == "Traceback (most recent call last):\n"


@pytest.mark.parametrize(
    "filename,split,config,raises",
    [
        ("config/builder-split.parquet", "split", "config", False),
        ("config/builder-with-dashes-split.parquet", "split", "config", False),
        ("config/builder-split-00000-of-00001.parquet", "split", "config", False),
        ("config/builder-with-dashes-split-00000-of-00001.parquet", "split", "config", False),
        ("config/builder-split.with.dots-00000-of-00001.parquet", "split.with.dots", "config", False),
        (
            "config/builder-with-dashes-caveat-asplitwithdashesisnotsupported-00000-of-00001.parquet",
            "asplitwithdashesisnotsupported",
            "config",
            False,
        ),
        ("builder-split-00000-of-00001.parquet", "split", "config", True),
        ("plain_text/openwebtext-10k-train.parquet", "train", "plain_text", False),
        ("plain_text/openwebtext-10k-train-00000-of-00001.parquet", "train", "plain_text", False),
    ],
)
def test_parse_repo_filename(filename: str, split: str, config: str, raises: bool) -> None:
    if raises:
        with pytest.raises(Exception):
            parse_repo_filename(filename)
    else:
        assert parse_repo_filename(filename) == (config, split)
