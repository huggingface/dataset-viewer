# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import io
import os
from contextlib import contextmanager
from dataclasses import replace
from fnmatch import fnmatch
from http import HTTPStatus
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Set, TypedDict
from unittest.mock import patch

import datasets.builder
import datasets.info
import pandas as pd
import pytest
import requests
from datasets import Audio, Features, Image, Value, load_dataset_builder
from huggingface_hub.hf_api import CommitOperationAdd, HfApi
from libcommon.dataset import get_dataset_info_for_supported_datasets
from libcommon.exceptions import (
    CustomError,
    DatasetInBlockListError,
    DatasetManualDownloadError,
)
from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.queue import Queue
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import CachedArtifactError, upsert_response
from libcommon.utils import JobInfo, JobParams, Priority

from worker.config import AppConfig
from worker.dtos import CompleteJobResult
from worker.job_manager import JobManager
from worker.job_runners.config.parquet_and_info import (
    ConfigParquetAndInfoJobRunner,
    _is_too_big_from_datasets,
    _is_too_big_from_external_data_files,
    _is_too_big_from_hub,
    create_commits,
    get_delete_operations,
    get_writer_batch_size,
    parse_repo_filename,
    raise_if_blocked,
    raise_if_requires_manual_download,
    stream_convert_to_parquet,
)
from worker.job_runners.dataset.config_names import DatasetConfigNamesJobRunner
from worker.resources import LibrariesResource

from ...constants import CI_HUB_ENDPOINT, CI_USER_TOKEN
from ...fixtures.hub import HubDatasetTest


@contextmanager
def blocked(app_config: AppConfig, repo_id: str) -> Iterator[None]:
    app_config.parquet_and_info.blocked_datasets.append(repo_id)
    yield
    app_config.parquet_and_info.blocked_datasets.remove(repo_id)


GetJobRunner = Callable[[str, str, AppConfig], ConfigParquetAndInfoJobRunner]


@pytest.fixture
def get_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
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
                    "input_type": "dataset",
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


def assert_content_is_equal(content: Any, expected: Any) -> None:
    print(content)
    assert set(content) == {"parquet_files", "dataset_info"}, content
    assert content["parquet_files"] == expected["parquet_files"], content
    assert len(content["dataset_info"]) == len(expected["dataset_info"]), content
    content_value = content["dataset_info"]
    expected_value = expected["dataset_info"]
    assert set(content_value.keys()) == set(expected_value.keys()), content
    for key in content_value.keys():
        if key != "download_checksums":
            assert content_value[key] == expected_value[key], content


def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    hub_responses_public: HubDatasetTest,
) -> None:
    dataset = hub_responses_public["name"]
    config = hub_responses_public["config_names_response"]["config_names"][0]["config"]
    upsert_response(
        "dataset-config-names",
        dataset=dataset,
        http_status=HTTPStatus.OK,
        content=hub_responses_public["config_names_response"],
    )
    job_runner = get_job_runner(dataset, config, app_config)
    response = job_runner.compute()
    assert response
    content = response.content
    assert content
    assert len(content["parquet_files"]) == 1
    assert_content_is_equal(content, hub_responses_public["parquet_and_info_response"])


def test_compute_legacy_configs(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    hub_public_legacy_configs: str,
) -> None:
    app_config = replace(app_config, parquet_and_info=replace(app_config.parquet_and_info, max_dataset_size=20_000))

    dataset_name = hub_public_legacy_configs
    original_configs = {"first", "second"}
    upsert_response(
        kind="dataset-config-names",
        dataset=hub_public_legacy_configs,
        http_status=HTTPStatus.OK,
        content={
            "config_names": [
                {"dataset": hub_public_legacy_configs, "config": "first"},
                {"dataset": hub_public_legacy_configs, "config": "second"},
            ],
        },
    )
    # first compute and push parquet files for each config for dataset with script with two configs
    for config in original_configs:
        job_runner = get_job_runner(dataset_name, config, app_config)
        assert job_runner.compute()
    hf_api = HfApi(endpoint=CI_HUB_ENDPOINT, token=CI_USER_TOKEN)
    dataset_info = hf_api.dataset_info(
        repo_id=hub_public_legacy_configs, revision=app_config.parquet_and_info.target_revision, files_metadata=False
    )
    repo_files = {f.rfilename for f in dataset_info.siblings}
    # assert that there are only parquet files for dataset's configs and ".gitattributes" in a repo
    # (no files from 'main')
    assert ".gitattributes" in repo_files
    assert all(
        fnmatch(file, "first/*.parquet") or fnmatch(file, "second/*.parquet")
        for file in repo_files.difference({".gitattributes"})
    )
    orig_repo_configs = {f.rfilename.split("/")[0] for f in dataset_info.siblings if f.rfilename.endswith(".parquet")}
    # assert that both configs are pushed (push of second config didn't delete first config's files)
    assert len(orig_repo_configs) == 2
    assert orig_repo_configs == original_configs
    # then change the set of dataset configs (remove "second")
    upsert_response(
        kind="dataset-config-names",
        dataset=hub_public_legacy_configs,
        http_status=HTTPStatus.OK,
        content={
            "config_names": [
                {"dataset": hub_public_legacy_configs, "config": "first"},
            ],
        },
    )
    job_runner = get_job_runner(dataset_name, "first", app_config)
    assert job_runner.compute()
    dataset_info = hf_api.dataset_info(
        repo_id=hub_public_legacy_configs, revision=app_config.parquet_and_info.target_revision, files_metadata=False
    )
    updated_repo_files = {f.rfilename for f in dataset_info.siblings}
    # assert that legacy config is removed from the repo
    # and there are only files for config that was just pushed and .gitattributes
    assert ".gitattributes" in updated_repo_files
    assert all(fnmatch(file, "first/*") for file in updated_repo_files.difference({".gitattributes"}))
    updated_repo_configs = {
        f.rfilename.split("/")[0] for f in dataset_info.siblings if f.rfilename.endswith(".parquet")
    }
    assert len(updated_repo_configs) == 1
    assert updated_repo_configs == {"first"}


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


def test_raise_if_requires_manual_download(hub_public_manual_download: str, app_config: AppConfig) -> None:
    builder = load_dataset_builder(hub_public_manual_download)
    with pytest.raises(DatasetManualDownloadError):
        raise_if_requires_manual_download(
            builder=builder,
            hf_endpoint=app_config.common.hf_endpoint,
            hf_token=app_config.common.hf_token,
        )


@pytest.mark.parametrize(
    "name,expected",
    [("public", False), ("big", True)],
)
def test__is_too_big_from_hub(
    hub_public_csv: str,
    hub_public_big: str,
    name: str,
    expected: bool,
    app_config: AppConfig,
) -> None:
    dataset = hub_public_csv if name == "public" else hub_public_big
    dataset_info = get_dataset_info_for_supported_datasets(
        dataset=dataset,
        hf_endpoint=app_config.common.hf_endpoint,
        hf_token=app_config.common.hf_token,
        revision="main",
        files_metadata=True,
    )
    assert (
        _is_too_big_from_hub(dataset_info=dataset_info, max_dataset_size=app_config.parquet_and_info.max_dataset_size)
        == expected
    )


@pytest.mark.parametrize(
    "name,expected",
    [("public", False), ("big", True)],
)
def test__is_too_big_from_datasets(
    hub_public_csv: str,
    hub_public_big: str,
    name: str,
    expected: bool,
    app_config: AppConfig,
) -> None:
    dataset = hub_public_csv if name == "public" else hub_public_big
    builder = load_dataset_builder(dataset)
    assert (
        _is_too_big_from_datasets(
            info=builder.info,
            max_dataset_size=app_config.parquet_and_info.max_dataset_size,
        )
        == expected
    )


@pytest.mark.parametrize(
    "max_dataset_size,max_external_data_files,expected",
    [
        (None, None, False),
        (10, None, True),
    ],
)
def test__is_too_big_external_files(
    external_files_dataset_builder: "datasets.builder.DatasetBuilder",
    expected: bool,
    max_dataset_size: Optional[int],
    max_external_data_files: Optional[int],
    app_config: AppConfig,
) -> None:
    max_dataset_size = max_dataset_size or app_config.parquet_and_info.max_dataset_size
    max_external_data_files = max_external_data_files or app_config.parquet_and_info.max_external_data_files
    assert (
        _is_too_big_from_external_data_files(
            builder=external_files_dataset_builder,
            hf_token=app_config.common.hf_token,
            max_dataset_size=max_dataset_size,
            max_external_data_files=max_external_data_files,
        )
        == expected
    )


@pytest.mark.parametrize(
    "max_dataset_size,max_external_data_files,expected",
    [
        (None, None, False),
        (None, 1, True),
    ],
)
def test_raise_if_too_many_external_files(
    external_files_dataset_builder: "datasets.builder.DatasetBuilder",
    expected: bool,
    max_dataset_size: Optional[int],
    max_external_data_files: Optional[int],
    app_config: AppConfig,
) -> None:
    max_dataset_size = max_dataset_size or app_config.parquet_and_info.max_dataset_size
    max_external_data_files = max_external_data_files or app_config.parquet_and_info.max_external_data_files
    assert (
        _is_too_big_from_external_data_files(
            builder=external_files_dataset_builder,
            hf_token=app_config.common.hf_token,
            max_dataset_size=max_dataset_size,
            max_external_data_files=max_external_data_files,
        )
        == expected
    )


def test_supported_if_big_parquet(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    hub_reponses_big: HubDatasetTest,
) -> None:
    # Not in the list of supported datasets and bigger than the maximum size
    # but still supported since it's made of parquet files
    # dataset = hub_public_big
    dataset = hub_reponses_big["name"]
    config = hub_reponses_big["config_names_response"]["config_names"][0]["config"]
    upsert_response(
        kind="dataset-config-names",
        dataset=dataset,
        http_status=HTTPStatus.OK,
        content=hub_reponses_big["config_names_response"],
    )
    job_runner = get_job_runner(dataset, config, app_config)
    response = job_runner.compute()
    assert response
    content = response.content
    assert content
    assert len(content["parquet_files"]) == 1
    assert_content_is_equal(content, hub_reponses_big["parquet_and_info_response"])


def test_partially_supported_if_big_non_parquet(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    hub_reponses_big_csv: HubDatasetTest,
) -> None:
    # Not in the list of supported datasets and bigger than the maximum size
    # dataset = hub_public_big_csv
    dataset = hub_reponses_big_csv["name"]
    config = hub_reponses_big_csv["config_names_response"]["config_names"][0]["config"]
    upsert_response(
        kind="dataset-config-names",
        dataset=dataset,
        http_status=HTTPStatus.OK,
        content=hub_reponses_big_csv["config_names_response"],
    )
    job_runner = get_job_runner(dataset, config, app_config)
    from datasets.packaged_modules.csv.csv import CsvConfig

    # Set a small chunk size to yield more than one Arrow Table in _generate_tables
    # to be able to stop the generation mid-way.
    with patch.object(CsvConfig, "pd_read_csv_kwargs", {"chunksize": 10}):
        response = job_runner.compute()
    assert response
    content = response.content
    assert content
    assert len(content["parquet_files"]) == 1
    assert_content_is_equal(content, hub_reponses_big_csv["parquet_and_info_response"])
    # dataset is partially generated
    assert content["parquet_files"][0]["size"] < app_config.parquet_and_info.max_dataset_size


def test_supported_if_gated(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    hub_reponses_gated: HubDatasetTest,
) -> None:
    # Access must be granted
    dataset = hub_reponses_gated["name"]
    config = hub_reponses_gated["config_names_response"]["config_names"][0]["config"]
    upsert_response(
        "dataset-config-names",
        dataset=dataset,
        http_status=HTTPStatus.OK,
        content=hub_reponses_gated["config_names_response"],
    )
    job_runner = get_job_runner(dataset, config, app_config)
    response = job_runner.compute()
    assert response
    assert response.content


def test_blocked(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    hub_reponses_jsonl: HubDatasetTest,
) -> None:
    # In the list of blocked datasets
    with blocked(app_config, repo_id=hub_reponses_jsonl["name"]):
        dataset = hub_reponses_jsonl["name"]
        config = hub_reponses_jsonl["config_names_response"]["config_names"][0]["config"]
        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            http_status=HTTPStatus.OK,
            content=hub_reponses_jsonl["config_names_response"],
        )
        job_runner = get_job_runner(dataset, config, app_config)
        with pytest.raises(CustomError) as e:
            job_runner.compute()
        assert e.typename == "DatasetInBlockListError"


@pytest.mark.parametrize(
    "name",
    ["public", "audio", "gated"],
)
def test_compute_splits_response_simple_csv_ok(
    hub_responses_public: HubDatasetTest,
    hub_reponses_audio: HubDatasetTest,
    hub_reponses_gated: HubDatasetTest,
    get_job_runner: GetJobRunner,
    name: str,
    app_config: AppConfig,
    data_df: pd.DataFrame,
) -> None:
    hub_datasets = {"public": hub_responses_public, "audio": hub_reponses_audio, "gated": hub_reponses_gated}
    dataset = hub_datasets[name]["name"]
    config = hub_datasets[name]["config_names_response"]["config_names"][0]["config"]
    upsert_response(
        "dataset-config-names",
        dataset=dataset,
        http_status=HTTPStatus.OK,
        content=hub_datasets[name]["config_names_response"],
    )
    expected_parquet_and_info_response = hub_datasets[name]["parquet_and_info_response"]
    job_runner = get_job_runner(dataset, config, app_config)
    result = job_runner.compute().content
    assert_content_is_equal(result, expected_parquet_and_info_response)

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
        ("private", "DatasetNotFoundError", None),
    ],
)
def test_compute_splits_response_simple_csv_error(
    hub_reponses_private: HubDatasetTest,
    get_job_runner: GetJobRunner,
    name: str,
    error_code: str,
    cause: str,
    app_config: AppConfig,
) -> None:
    dataset = hub_reponses_private["name"]
    config_names_response = hub_reponses_private["config_names_response"]
    config = config_names_response["config_names"][0]["config"] if config_names_response else None
    upsert_response(
        "dataset-config-names",
        dataset=dataset,
        http_status=HTTPStatus.OK,
        content=hub_reponses_private["config_names_response"],
    )
    job_runner = get_job_runner(dataset, config, app_config)
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
    "name,error_code,cause",
    [
        ("public", "CachedResponseNotFound", None),  # no cache for dataset-config-names -> CachedResponseNotFound
    ],
)
def test_compute_splits_response_simple_csv_error_2(
    hub_responses_public: HubDatasetTest,
    get_job_runner: GetJobRunner,
    name: str,
    error_code: str,
    cause: str,
    app_config: AppConfig,
) -> None:
    dataset = hub_responses_public["name"]
    config_names_response = hub_responses_public["config_names_response"]
    config = config_names_response["config_names"][0]["config"] if config_names_response else None
    job_runner = get_job_runner(dataset, config, app_config)
    with pytest.raises(CachedArtifactError):
        job_runner.compute()


@pytest.mark.parametrize(
    "upstream_status,upstream_content,exception_name",
    [
        (HTTPStatus.NOT_FOUND, {"error": "error"}, "CachedArtifactError"),
        (HTTPStatus.OK, {"not_config_names": "wrong_format"}, "PreviousStepFormatError"),
        (HTTPStatus.OK, {"config_names": "not a list"}, "PreviousStepFormatError"),
    ],
)
def test_previous_step_error(
    get_job_runner: GetJobRunner,
    upstream_status: HTTPStatus,
    upstream_content: Any,
    exception_name: str,
    hub_responses_public: HubDatasetTest,
    app_config: AppConfig,
) -> None:
    dataset = hub_responses_public["name"]
    config = hub_responses_public["config_names_response"]["config_names"][0]["config"]
    job_runner = get_job_runner(dataset, config, app_config)
    upsert_response(
        "dataset-config-names",
        dataset=dataset,
        http_status=upstream_status,
        content=upstream_content,
    )
    with pytest.raises(Exception) as exc_info:
        job_runner.compute()
    assert exc_info.typename == exception_name


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


@pytest.mark.parametrize(
    "ds_info, has_big_chunks",
    [
        (datasets.info.DatasetInfo(), False),
        (datasets.info.DatasetInfo(features=Features({"text": Value("string")})), False),
        (datasets.info.DatasetInfo(features=Features({"image": Image()})), True),
        (datasets.info.DatasetInfo(features=Features({"audio": Audio()})), True),
        (datasets.info.DatasetInfo(features=Features({"nested": [{"image": Image()}]})), True),
        (datasets.info.DatasetInfo(features=Features({"blob": Value("binary")})), True),
    ],
)
def test_get_writer_batch_size(ds_info: datasets.info.DatasetInfo, has_big_chunks: bool) -> None:
    assert get_writer_batch_size(ds_info) == (100 if has_big_chunks else None)


@pytest.mark.parametrize(
    "max_operations_per_commit,use_parent_commit,expected_num_commits",
    [(2, False, 1), (1, False, 2), (2, True, 1), (1, True, 2)],
)
def test_create_commits(
    hub_public_legacy_configs: str, max_operations_per_commit: int, use_parent_commit: bool, expected_num_commits: int
) -> None:
    NUM_FILES = 2
    repo_id = hub_public_legacy_configs
    hf_api = HfApi(endpoint=CI_HUB_ENDPOINT, token=CI_USER_TOKEN)
    if use_parent_commit:
        target_dataset_info = hf_api.dataset_info(repo_id=repo_id, files_metadata=False)
        parent_commit = target_dataset_info.sha
    else:
        parent_commit = None
    directory = f".test_create_commits_{max_operations_per_commit}_{use_parent_commit}"
    operations: List[CommitOperationAdd] = [
        CommitOperationAdd(path_in_repo=f"{directory}/file{i}.txt", path_or_fileobj=f"content{i}".encode("UTF-8"))
        for i in range(NUM_FILES)
    ]
    commit_infos = create_commits(
        hf_api=hf_api,
        repo_id=repo_id,
        operations=operations,
        commit_message="test",
        max_operations_per_commit=max_operations_per_commit,
        parent_commit=parent_commit,
    )
    assert len(commit_infos) == expected_num_commits
    # check that the files were created
    filenames = hf_api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    for i in range(NUM_FILES):
        assert f"{directory}/file{i}.txt" in filenames


GetDatasetConfigNamesJobRunner = Callable[[str, AppConfig], DatasetConfigNamesJobRunner]


@pytest.fixture
def get_dataset_config_names_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetDatasetConfigNamesJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
    ) -> DatasetConfigNamesJobRunner:
        processing_step_name = DatasetConfigNamesJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            {
                processing_step_name: {
                    "input_type": "dataset",
                    "job_runner_version": DatasetConfigNamesJobRunner.get_job_runner_version(),
                }
            }
        )
        return DatasetConfigNamesJobRunner(
            job_info={
                "type": DatasetConfigNamesJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
                    "config": None,
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


class JobRunnerArgs(TypedDict):
    dataset: str
    revision: str
    config: str
    app_config: AppConfig
    tmp_path: Path


def launch_job_runner(job_runner_args: JobRunnerArgs) -> CompleteJobResult:
    config = job_runner_args["config"]
    dataset = job_runner_args["dataset"]
    revision = job_runner_args["revision"]
    app_config = job_runner_args["app_config"]
    tmp_path = job_runner_args["tmp_path"]
    job_runner = ConfigParquetAndInfoJobRunner(
        job_info=JobInfo(
            job_id=f"job_{config}",
            type="config-parquet-and-info",
            params=JobParams(dataset=dataset, revision=revision, config=config, split=None),
            priority=Priority.NORMAL,
        ),
        app_config=app_config,
        processing_step=ProcessingStep(
            name="config-parquet-and-info",
            input_type="config",
            job_runner_version=ConfigParquetAndInfoJobRunner.get_job_runner_version(),
        ),
        hf_datasets_cache=tmp_path,
    )
    return job_runner.compute()


def test_concurrency(
    hub_public_n_configs: str,
    app_config: AppConfig,
    tmp_path: Path,
    get_dataset_config_names_job_runner: GetDatasetConfigNamesJobRunner,
    queue_mongo_resource: QueueMongoResource,
    cache_mongo_resource: CacheMongoResource,
) -> None:
    """
    Test that multiple job runners (to compute config-parquet-and-info) can run in parallel,
    without having conflicts when sending commits to the Hub.
    For this test, we need a lot of configs for the same dataset (say 20) and one job runner for each.
    Ideally we would try for both quick and slow jobs.
    """
    repo_id = hub_public_n_configs
    hf_api = HfApi(endpoint=CI_HUB_ENDPOINT, token=CI_USER_TOKEN)
    revision = hf_api.dataset_info(repo_id=repo_id, files_metadata=False).sha
    if revision is None:
        raise ValueError(f"Could not find revision for dataset {repo_id}")

    # fill the cache for the step dataset-config-names, required by the job_runner
    # it's a lot of code 😅
    job_info = JobInfo(
        job_id="not_used",
        type="dataset-config-names",
        params=JobParams(dataset=repo_id, revision=revision, config=None, split=None),
        priority=Priority.NORMAL,
    )
    queue = Queue()
    queue.create_jobs([job_info])
    job_info = queue.start_job(job_types_only=["dataset-config-names"])
    job_manager = JobManager(
        job_info=job_info,
        app_config=app_config,
        processing_graph=ProcessingGraph(
            {
                "dataset-config-names": {
                    "input_type": "dataset",
                    "provides_dataset_config_names": True,
                    "job_runner_version": DatasetConfigNamesJobRunner.get_job_runner_version(),
                }
            }
        ),
        job_runner=get_dataset_config_names_job_runner(repo_id, app_config),
    )
    job_result = job_manager.run_job()
    job_manager.finish(job_result=job_result)
    if not job_result["output"]:
        raise ValueError("Could not get config names")
    configs = [str(config_name["config"]) for config_name in job_result["output"]["content"]["config_names"]]

    # launch the job runners
    NUM_JOB_RUNNERS = 10
    with Pool(NUM_JOB_RUNNERS) as pool:
        pool.map(
            launch_job_runner,
            [
                JobRunnerArgs(
                    dataset=repo_id, revision=revision, config=config, app_config=app_config, tmp_path=tmp_path
                )
                for config in configs
            ],
        )


@pytest.mark.parametrize(
    "parquet_files,all_repo_files,config_names,config,deleted_files",
    [
        (
            set(),
            {"dummy", "c1/dummy", "c1/0.parquet", "c2/0.parquet", "c1/index.duckdb"},
            {"c1", "c2"},
            "c1",
            {"dummy", "c1/dummy", "c1/0.parquet"},
        ),
        (
            {"c1/0.parquet"},
            {"dummy", "c1/dummy", "c1/0.parquet", "c2/0.parquet", "c1/index.duckdb"},
            {"c1", "c2"},
            "c1",
            {"dummy", "c1/dummy"},
        ),
    ],
)
def test_get_delete_operations(
    parquet_files: Set[str], all_repo_files: Set[str], config_names: Set[str], config: str, deleted_files: Set[str]
) -> None:
    parquet_operations = [
        CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=b"") for path_in_repo in parquet_files
    ]
    delete_operations = get_delete_operations(
        parquet_operations=parquet_operations, all_repo_files=all_repo_files, config_names=config_names, config=config
    )
    assert set(delete_operation.path_in_repo for delete_operation in delete_operations) == deleted_files


@pytest.mark.parametrize(
    "max_dataset_size,expected_num_shards",
    [
        (1, 1),
        (100, 2),
        (1000, 10),
    ],
)
def test_stream_convert_to_parquet(
    csv_path: str, max_dataset_size: int, expected_num_shards: int, tmp_path: Path
) -> None:
    num_data_files = 10
    builder = load_dataset_builder(
        "csv",
        data_files={"train": [csv_path] * num_data_files},
        cache_dir=str(tmp_path / f"test_stream_convert_to_parquet-{max_dataset_size=}"),
    )
    with patch("worker.job_runners.config.parquet_and_info.get_writer_batch_size", lambda ds_config_info: 1):
        with patch.object(datasets.config, "MAX_SHARD_SIZE", 1):
            parquet_operations, full = stream_convert_to_parquet(builder, max_dataset_size=max_dataset_size)
    num_shards = len(parquet_operations)
    assert num_shards == expected_num_shards
    assert full == (expected_num_shards == num_data_files)
    assert all(isinstance(op.path_or_fileobj, str) for op in parquet_operations)
    parquet_filenames = sorted(os.path.basename(op.path_or_fileobj) for op in parquet_operations)
    if num_shards == 1:
        assert parquet_filenames[0] == "csv-train.parquet"
    else:
        assert all(
            parquet_filename == f"csv-train-{shard_idx:05d}-of-{num_shards:05d}.parquet"
            for shard_idx, parquet_filename in enumerate(parquet_filenames)
        )
