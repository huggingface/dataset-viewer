# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from collections.abc import Callable, Iterator
from http import HTTPStatus
from typing import Any

import fsspec
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from libcommon.dtos import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import CachedArtifactError, upsert_response
from pytest import TempPathFactory

from worker.config import AppConfig
from worker.job_runners.dataset.loading_tags import (
    DatasetLoadingTagsJobRunner,
    get_builder_configs_with_simplified_data_files,
)
from worker.resources import LibrariesResource

from ..utils import REVISION_NAME, UpstreamResponse


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, AppConfig], DatasetLoadingTagsJobRunner]

PARQUET_DATASET = "parquet-dataset"
WEBDATASET_DATASET = "webdataset-dataset"
ERROR_DATASET = "error-dataset"

UPSTREAM_RESPONSE_INFO_PARQUET: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=PARQUET_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={"dataset_info": {"default": {"config_name": "default", "builder_name": "parquet"}}, "partial": False},
    progress=1.0,
)
UPSTREAM_RESPONSE_INFO_WEBDATASET: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=WEBDATASET_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={"dataset_info": {"default": {"config_name": "default", "builder_name": "webdataset"}}, "partial": False},
    progress=1.0,
)
UPSTREAM_RESPONSE_INFD_ERROR: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=ERROR_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
    progress=0.0,
)
EXPECTED_PARQUET = (
    {
        "tags": ["croissant"],
        "libraries": [
            {
                "function": "Dataset",
                "language": "python",
                "library": "mlcroissant",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {"record_set": "default", "partial": False},
                        "code": (
                            "from mlcroissant "
                            "import Dataset\n"
                            "\n"
                            'ds = Dataset(jsonld="https://datasets-server.huggingface.co/croissant?dataset=parquet-dataset")\n'
                            'records = ds.records("default")'
                        ),
                    }
                ],
            },
            {
                "function": "load_dataset",
                "language": "python",
                "library": "datasets",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {},
                        "code": ('from datasets import load_dataset\n\nds = load_dataset("parquet-dataset")'),
                    }
                ],
            },
            {
                "function": "pd.read_parquet",
                "language": "python",
                "library": "pandas",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {
                            "splits": {
                                "test": "test.parquet",
                                "train": "train.parquet",
                            }
                        },
                        "code": (
                            "import pandas as pd\n"
                            "\n"
                            "splits = {'train': 'train.parquet', 'test': 'test.parquet'}\n"
                            'df = pd.read_parquet("hf://datasets/parquet-dataset/" + splits["train"])'
                        ),
                    }
                ],
            },
        ],
    },
    1.0,
)
EXPECTED_WEBDATASET = (
    {
        "tags": ["croissant"],
        "libraries": [
            {
                "function": "Dataset",
                "language": "python",
                "library": "mlcroissant",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {"record_set": "default", "partial": False},
                        "code": (
                            "from mlcroissant import Dataset\n"
                            "\n"
                            'ds = Dataset(jsonld="https://datasets-server.huggingface.co/croissant?dataset=webdataset-dataset")\n'
                            'records = ds.records("default")'
                        ),
                    }
                ],
            },
            {
                "function": "load_dataset",
                "language": "python",
                "library": "datasets",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {},
                        "code": ('from datasets import load_dataset\n\nds = load_dataset("webdataset-dataset")'),
                    }
                ],
            },
            {
                "function": "wds.WebDataset",
                "language": "python",
                "library": "webdataset",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {"splits": {"train": "**/*.tar"}},
                        "code": (
                            "import webdataset as wds\n"
                            "from huggingface_hub import HfFileSystem, get_token, hf_hub_url\n"
                            "\n"
                            "fs = HfFileSystem()\n"
                            'files = [fs.resolve_path(path) for path in fs.glob("hf://datasets/webdataset-dataset/**/*.tar")]\n'
                            'urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]\n'
                            "urls = f\"pipe: curl -s -L -H 'Authorization:Bearer {get_token()}' {'::'.join(urls)}\"\n"
                            "\n"
                            "ds = wds.WebDataset(urls).decode()"
                        ),
                    }
                ],
            },
        ],
    },
    1.0,
)


@pytest.fixture()
def mock_hffs(tmp_path_factory: TempPathFactory) -> Iterator[fsspec.AbstractFileSystem]:
    hf = tmp_path_factory.mktemp("hf")

    (hf / "datasets" / PARQUET_DATASET).mkdir(parents=True)
    (hf / "datasets" / PARQUET_DATASET / "train.parquet").touch()
    (hf / "datasets" / PARQUET_DATASET / "test.parquet").touch()

    (hf / "datasets" / WEBDATASET_DATASET).mkdir(parents=True)
    (hf / "datasets" / WEBDATASET_DATASET / "0000.tar").touch()
    (hf / "datasets" / WEBDATASET_DATASET / "0001.tar").touch()
    (hf / "datasets" / WEBDATASET_DATASET / "0002.tar").touch()
    (hf / "datasets" / WEBDATASET_DATASET / "0003.tar").touch()

    class MockHfFileSystem(DirFileSystem):  # type: ignore[misc]
        protocol = "hf"

        def __init__(self, path: str = str(hf), target_protocol: str = "local", **kwargs: dict[str, Any]) -> None:
            super().__init__(path=path, target_protocol=target_protocol, **kwargs)

    HfFileSystem = fsspec.get_filesystem_class("hf")
    fsspec.register_implementation("hf", MockHfFileSystem, clobber=True)
    yield MockHfFileSystem()
    fsspec.register_implementation("hf", HfFileSystem, clobber=True)


@pytest.fixture
def get_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
    ) -> DatasetLoadingTagsJobRunner:
        return DatasetLoadingTagsJobRunner(
            job_info={
                "type": DatasetLoadingTagsJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "config": None,
                    "split": None,
                    "revision": REVISION_NAME,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 20,
            },
            app_config=app_config,
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,upstream_responses,expected",
    [
        (
            PARQUET_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_PARQUET,
            ],
            EXPECTED_PARQUET,
        ),
        (
            WEBDATASET_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_WEBDATASET,
            ],
            EXPECTED_WEBDATASET,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    mock_hffs: fsspec.AbstractFileSystem,
    dataset: str,
    upstream_responses: list[UpstreamResponse],
    expected: Any,
) -> None:
    for upstream_response in upstream_responses:
        upsert_response(**upstream_response)
    job_runner = get_job_runner(dataset, app_config)
    compute_result = job_runner.compute()
    assert compute_result.content == expected[0]
    assert compute_result.progress == expected[1]


@pytest.mark.parametrize(
    "dataset,upstream_responses,expectation",
    [
        (
            ERROR_DATASET,
            [
                UPSTREAM_RESPONSE_INFD_ERROR,
            ],
            pytest.raises(CachedArtifactError),
        )
    ],
)
def test_compute_error(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    mock_hffs: fsspec.AbstractFileSystem,
    dataset: str,
    upstream_responses: list[UpstreamResponse],
    expectation: Any,
) -> None:
    for upstream_response in upstream_responses:
        upsert_response(**upstream_response)
    job_runner = get_job_runner(dataset, app_config)
    with expectation:
        job_runner.compute()


@pytest.mark.real_dataset
@pytest.mark.parametrize(
    "dataset,module_name,expected_simplified_data_files",
    [
        (
            "Anthropic/hh-rlhf",
            "json",
            {"default": {"test": ["**/*/test.jsonl.gz"], "train": ["**/*/train.jsonl.gz"]}},
        ),
        ("laion/conceptual-captions-12m-webdataset", "webdataset", {"default": {"train": ["**/*.tar"]}}),
        (
            "tatsu-lab/alpaca",
            "parquet",
            {"default": {"train": ["data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"]}},
        ),
        (
            "christopherthompson81/quant_exploration",
            "json",
            {"default": {"train": ["quant_exploration.json"]}},
        ),
        (
            "teknium/openhermes",
            "json",
            {"default": {"train": ["openhermes.json"]}},
        ),
        (
            "cnn_dailymail",
            "parquet",
            {
                "1.0.0": {"test": ["1.0.0/test-*"], "train": ["1.0.0/train-*"], "validation": ["1.0.0/validation-*"]},
                "2.0.0": {"test": ["2.0.0/test-*"], "train": ["2.0.0/train-*"], "validation": ["2.0.0/validation-*"]},
                "3.0.0": {"test": ["3.0.0/test-*"], "train": ["3.0.0/train-*"], "validation": ["3.0.0/validation-*"]},
            },
        ),
        (
            "lmsys/toxic-chat",
            "csv",
            {
                "toxicchat0124": {
                    "test": ["data/0124/toxic-chat_annotation_test.csv"],
                    "train": ["data/0124/toxic-chat_annotation_train.csv"],
                },
                "toxicchat1123": {
                    "test": ["data/1123/toxic-chat_annotation_test.csv"],
                    "train": ["data/1123/toxic-chat_annotation_train.csv"],
                },
            },
        ),
    ],
)
def test_simplify_data_files_patterns(
    use_hub_prod_endpoint: pytest.MonkeyPatch,
    dataset: str,
    module_name: str,
    expected_simplified_data_files: dict[str, dict[str, list[str]]],
) -> None:
    configs = get_builder_configs_with_simplified_data_files(dataset, module_name=module_name)
    simplified_data_files: dict[str, dict[str, list[str]]] = {config.name: config.data_files for config in configs}
    assert simplified_data_files == expected_simplified_data_files
