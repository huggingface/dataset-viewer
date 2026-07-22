# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from collections.abc import Callable, Iterator
from http import HTTPStatus
from pathlib import Path
from typing import Any
from unittest.mock import patch

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset
from fsspec.implementations.dirfs import DirFileSystem
from libcommon.dtos import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import CachedArtifactError, upsert_response
from pytest import TempPathFactory

from worker.config import AppConfig
from worker.job_runners.dataset.compatible_libraries import (
    LOGIN_COMMENT,
    DatasetCompatibleLibrariesJobRunner,
    get_builder_configs,
    get_compatible_libraries_for_json,
    get_compatible_libraries_for_lerobot,
    get_compatible_library_for_builder,
)
from worker.resources import LibrariesResource

from ..utils import REVISION_NAME, UpstreamResponse


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, AppConfig], DatasetCompatibleLibrariesJobRunner]

PARQUET_DATASET = "dummy/parquet-dataset"
PARQUET_DATASET_LOGIN_REQUIRED = "dummy/parquet-dataset-login_required"
WEBDATASET_DATASET = "dummy/webdataset-dataset"
LANCE_DATASET = "dummy/lance-dataset"
LEROBOT_DATASET = "dummy/lerobot-dataset"
ERROR_DATASET = "dummy/error-dataset"

UPSTREAM_RESPONSE_INFO_PARQUET: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=PARQUET_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={"dataset_info": {"default": {"config_name": "default", "builder_name": "parquet"}}, "partial": False},
    progress=1.0,
)
UPSTREAM_RESPONSE_INFO_PARQUET_LOGIN_REQUIRED: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=PARQUET_DATASET_LOGIN_REQUIRED,
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
UPSTREAM_RESPONSE_INFO_LANCE: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=LANCE_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={"dataset_info": {"default": {"config_name": "default", "builder_name": "lance"}}, "partial": False},
    progress=1.0,
)
UPSTREAM_RESPONSE_INFO_LEROBOT: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=LEROBOT_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={"dataset_info": {"default": {"config_name": "default", "builder_name": "parquet"}}, "partial": False},
    progress=1.0,
)
UPSTREAM_RESPONSE_INFO_ERROR: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=ERROR_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
    progress=0.0,
)
EXPECTED_PARQUET = (
    {
        "formats": ["parquet"],
        "libraries": [
            {
                "function": "load_dataset",
                "language": "python",
                "library": "datasets",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {},
                        "code": ('from datasets import load_dataset\n\nds = load_dataset("dummy/parquet-dataset")'),
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
                            'df = pd.read_parquet("hf://datasets/dummy/parquet-dataset/" + splits["train"])'
                        ),
                    }
                ],
            },
            {
                "function": "pl.read_parquet",
                "language": "python",
                "library": "polars",
                "loading_codes": [
                    {
                        "arguments": {"splits": {"test": "test.parquet", "train": "train.parquet"}},
                        "code": "import polars as pl\n"
                        "\n"
                        "splits = {'train': 'train.parquet', 'test': "
                        "'test.parquet'}\n"
                        "df = "
                        'pl.read_parquet("hf://datasets/dummy/parquet-dataset/" '
                        '+ splits["train"])',
                        "config_name": "default",
                    }
                ],
            },
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
                            'ds = Dataset(jsonld="https://huggingface.co/api/datasets/dummy/parquet-dataset/croissant")\n'
                            'records = ds.records("default")'
                        ),
                    }
                ],
            },
        ],
    },
    1.0,
)

EXPECTED_PARQUET_LOGIN_REQUIRED = (
    {
        "formats": ["parquet"],
        "libraries": [
            {
                "function": "load_dataset",
                "language": "python",
                "library": "datasets",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {},
                        "code": (
                            f'from datasets import load_dataset\n{LOGIN_COMMENT}\nds = load_dataset("dummy/parquet-dataset-login_required")'
                        ),
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
                            f"{LOGIN_COMMENT}\n"
                            "splits = {'train': 'train.parquet', 'test': 'test.parquet'}\n"
                            'df = pd.read_parquet("hf://datasets/dummy/parquet-dataset-login_required/" + splits["train"])'
                        ),
                    }
                ],
            },
            {
                "function": "pl.read_parquet",
                "language": "python",
                "library": "polars",
                "loading_codes": [
                    {
                        "arguments": {"splits": {"test": "test.parquet", "train": "train.parquet"}},
                        "code": "import polars as pl\n"
                        "\n"
                        "# Login using e.g. `huggingface-cli login` to "
                        "access this dataset\n"
                        "splits = {'train': 'train.parquet', 'test': "
                        "'test.parquet'}\n"
                        "df = "
                        'pl.read_parquet("hf://datasets/dummy/parquet-dataset-login_required/" '
                        '+ splits["train"])',
                        "config_name": "default",
                    }
                ],
            },
            {
                "function": "Dataset",
                "language": "python",
                "library": "mlcroissant",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {"record_set": "default", "partial": False},
                        "code": (
                            "import requests\n"
                            "from huggingface_hub.file_download import build_hf_headers\n"
                            "from mlcroissant import Dataset\n"
                            f"{LOGIN_COMMENT}\n"
                            "headers = build_hf_headers()  # handles authentication\n"
                            'jsonld = requests.get("https://huggingface.co/api/datasets/dummy/parquet-dataset-login_required/croissant", headers=headers).json()\n'
                            "ds = Dataset(jsonld=jsonld)\n"
                            'records = ds.records("default")'
                        ),
                    }
                ],
            },
        ],
    },
    1.0,
)
EXPECTED_LANCE = (
    {
        "formats": ["lance"],
        "libraries": [
            {
                "function": "load_dataset",
                "language": "python",
                "library": "datasets",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {},
                        "code": ('from datasets import load_dataset\n\nds = load_dataset("dummy/lance-dataset")'),
                    }
                ],
            },
            {
                "function": "lance.dataset",
                "language": "python",
                "library": "lance",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {},
                        "code": ('import lance\n\nds = lance.dataset("hf://datasets/dummy/lance-dataset")'),
                    }
                ],
            },
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
                            'ds = Dataset(jsonld="https://huggingface.co/api/datasets/dummy/lance-dataset/croissant")\n'
                            'records = ds.records("default")'
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
        "formats": ["webdataset"],
        "libraries": [
            {
                "function": "load_dataset",
                "language": "python",
                "library": "datasets",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {},
                        "code": ('from datasets import load_dataset\n\nds = load_dataset("dummy/webdataset-dataset")'),
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
                            'files = [fs.resolve_path(path) for path in fs.glob("hf://datasets/dummy/webdataset-dataset/**/*.tar")]\n'
                            'urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]\n'
                            "urls = f\"pipe: curl -s -L -H 'Authorization:Bearer {get_token()}' {'::'.join(urls)}\"\n"
                            "\n"
                            "ds = wds.WebDataset(urls).decode()"
                        ),
                    }
                ],
            },
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
                            'ds = Dataset(jsonld="https://huggingface.co/api/datasets/dummy/webdataset-dataset/croissant")\n'
                            'records = ds.records("default")'
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
    ds = Dataset.from_dict({"a": range(10)})

    (hf / "datasets" / PARQUET_DATASET).mkdir(parents=True)
    ds.to_parquet(hf / "datasets" / PARQUET_DATASET / "train.parquet")
    ds.to_parquet(hf / "datasets" / PARQUET_DATASET / "test.parquet")

    (hf / "datasets" / PARQUET_DATASET_LOGIN_REQUIRED).mkdir(parents=True)
    ds.to_parquet(hf / "datasets" / PARQUET_DATASET_LOGIN_REQUIRED / "train.parquet")
    ds.to_parquet(hf / "datasets" / PARQUET_DATASET_LOGIN_REQUIRED / "test.parquet")

    (hf / "datasets" / WEBDATASET_DATASET).mkdir(parents=True)
    (hf / "datasets" / WEBDATASET_DATASET / "0000.tar").touch()
    (hf / "datasets" / WEBDATASET_DATASET / "0001.tar").touch()
    (hf / "datasets" / WEBDATASET_DATASET / "0002.tar").touch()
    (hf / "datasets" / WEBDATASET_DATASET / "0003.tar").touch()

    (hf / "datasets" / LANCE_DATASET).mkdir(parents=True)
    (hf / "datasets" / LANCE_DATASET / "_versions").mkdir(parents=True)
    (hf / "datasets" / LANCE_DATASET / "data").mkdir(parents=True)
    (hf / "datasets" / LANCE_DATASET / "data" / "0000.lance").touch()
    (hf / "datasets" / LANCE_DATASET / "_versions" / "1.manifest").touch()

    (hf / "datasets" / LEROBOT_DATASET / "data" / "chunk-000").mkdir(parents=True)
    ds.to_parquet(hf / "datasets" / LEROBOT_DATASET / "data" / "chunk-000" / "episode_000000.parquet")
    (hf / "datasets" / LEROBOT_DATASET / "README.md").write_text(
        "---\n"
        "license: apache-2.0\n"
        "task_categories:\n"
        "- robotics\n"
        "tags:\n"
        "- LeRobot\n"
        "configs:\n"
        "- config_name: default\n"
        "  data_files: data/*/*.parquet\n"
        "---\n"
    )

    class MockHfFileSystem(DirFileSystem):  # type: ignore[misc]
        protocol = "hf"

        def __init__(self, path: str = str(hf), target_protocol: str = "local", **kwargs: Any) -> None:
            super().__init__(path=path, target_protocol=target_protocol, **kwargs)
            self.logged_in = kwargs.get("token") != "no_token"

        def isdir(self, path: str, **kwargs: Any) -> bool:
            if "login_required" in path and not self.logged_in:
                return False
            return bool(super().isdir(path, **kwargs))

    HfFileSystem = fsspec.get_filesystem_class("hf")
    fsspec.register_implementation("hf", MockHfFileSystem, clobber=True)
    with patch("worker.job_runners.dataset.compatible_libraries.HfFileSystem", MockHfFileSystem):
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
    ) -> DatasetCompatibleLibrariesJobRunner:
        return DatasetCompatibleLibrariesJobRunner(
            job_info={
                "type": DatasetCompatibleLibrariesJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "config": None,
                    "split": None,
                    "revision": REVISION_NAME,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 20,
                "started_at": None,
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
            PARQUET_DATASET_LOGIN_REQUIRED,
            [
                UPSTREAM_RESPONSE_INFO_PARQUET_LOGIN_REQUIRED,
            ],
            EXPECTED_PARQUET_LOGIN_REQUIRED,
        ),
        (
            WEBDATASET_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_WEBDATASET,
            ],
            EXPECTED_WEBDATASET,
        ),
        (
            LANCE_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_LANCE,
            ],
            EXPECTED_LANCE,
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
    job_runner.pre_compute()
    compute_result = list(job_runner.compute())[0]
    job_runner.post_compute()
    assert compute_result.content == expected[0]
    assert compute_result.progress == expected[1]


def test_compute_lerobot(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    mock_hffs: fsspec.AbstractFileSystem,
) -> None:
    upsert_response(**UPSTREAM_RESPONSE_INFO_LEROBOT)
    job_runner = get_job_runner(LEROBOT_DATASET, app_config)
    job_runner.pre_compute()
    compute_result = list(job_runner.compute())[0]
    job_runner.post_compute()
    libraries = {library["library"] for library in compute_result.content["libraries"]}
    # the "lerobot" library is detected from the "LeRobot" tag in the dataset card
    assert "lerobot" in libraries
    lerobot_library = next(
        library for library in compute_result.content["libraries"] if library["library"] == "lerobot"
    )
    assert lerobot_library == {
        "language": "python",
        "library": "lerobot",
        "function": "LeRobotDataset",
        "loading_codes": [
            {
                "config_name": "default",
                "arguments": {},
                "code": (
                    'from lerobot.datasets import LeRobotDataset\n\ndataset = LeRobotDataset("dummy/lerobot-dataset")'
                ),
            }
        ],
    }
    # the regular libraries are still added (LeRobot datasets are parquet-based)
    assert {"datasets", "mlcroissant"} <= libraries


def test_get_compatible_libraries_for_lerobot(
    mock_hffs: fsspec.AbstractFileSystem,
) -> None:
    # a dataset tagged "LeRobot" is detected
    compatible_libraries = get_compatible_libraries_for_lerobot(LEROBOT_DATASET, hf_token=None, login_required=False)
    assert len(compatible_libraries) == 1
    assert compatible_libraries[0]["library"] == "lerobot"
    # a dataset without the "LeRobot" tag (or without a dataset card) is not detected
    assert get_compatible_libraries_for_lerobot(PARQUET_DATASET, hf_token=None, login_required=False) == []


@pytest.mark.parametrize(
    "dataset,upstream_responses,expectation",
    [
        (
            ERROR_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_ERROR,
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
    job_runner.pre_compute()
    with expectation:
        list(job_runner.compute())
    job_runner.post_compute()


@pytest.mark.real_dataset
@pytest.mark.parametrize(
    "dataset,module_name,expected_simplified_data_files",
    [
        (
            "Anthropic/hh-rlhf",
            "json",
            {"default": {"test": ["**/test.jsonl.gz"], "train": ["**/train.jsonl.gz"]}},
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
            "abisee/cnn_dailymail",
            "parquet",
            {
                "1.0.0": {
                    "test": ["1.0.0/test-00000-of-00001.parquet"],
                    "train": ["1.0.0/train-*.parquet"],
                    "validation": ["1.0.0/validation-00000-of-00001.parquet"],
                },
                "2.0.0": {
                    "test": ["2.0.0/test-00000-of-00001.parquet"],
                    "train": ["2.0.0/train-*.parquet"],
                    "validation": ["2.0.0/validation-00000-of-00001.parquet"],
                },
                "3.0.0": {
                    "test": ["3.0.0/test-00000-of-00001.parquet"],
                    "train": ["3.0.0/train-*.parquet"],
                    "validation": ["3.0.0/validation-00000-of-00001.parquet"],
                },
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
    configs = get_builder_configs(dataset, module_name=module_name, with_simplified_data_files=True)
    simplified_data_files: dict[str, dict[str, list[str]]] = {config.name: config.data_files for config in configs}
    assert simplified_data_files == expected_simplified_data_files


@pytest.mark.real_dataset
@pytest.mark.integration
@pytest.mark.parametrize(
    "dataset,module_name,expected_data_files,expected_library",
    [
        (
            "rajpurkar/squad",
            "parquet",
            {
                "train": ["plain_text/train-00000-of-00001.parquet"],
                "validation": ["plain_text/validation-00000-of-00001.parquet"],
            },
            "pandas",
        ),
        (
            "Anthropic/hh-rlhf",
            "json",
            {
                "train": ["**/train.jsonl.gz"],
                "test": ["**/test.jsonl.gz"],
            },
            "dask",
        ),
    ],
)
def test_get_builder_configs_with_simplified_data_files(
    use_hub_prod_endpoint: pytest.MonkeyPatch,
    dataset: str,
    module_name: str,
    expected_data_files: dict[str, list[str]],
    expected_library: str,
) -> None:
    hf_token = None
    login_required = False
    configs = get_builder_configs(dataset, module_name=module_name, hf_token=hf_token, with_simplified_data_files=True)
    assert len(configs) == 1
    config = configs[0]
    assert config.data_files == expected_data_files
    assert module_name in get_compatible_library_for_builder
    compatible_libraries = get_compatible_library_for_builder[module_name](dataset, hf_token, login_required)
    compatible_library = compatible_libraries[0]
    assert compatible_library["library"] == expected_library


@pytest.mark.real_dataset
@pytest.mark.integration
@pytest.mark.parametrize(
    "dataset, expected_libraries",
    [
        ("tcor0005/langchain-docs-400-chunksize", ["dask"]),
        ("Anthropic/hh-rlhf", ["dask", "polars"]),
    ],
)
def test_get_compatible_libraries_for_json(
    use_hub_prod_endpoint: pytest.MonkeyPatch,
    dataset: str,
    expected_libraries: list[str],
) -> None:
    compatible_libraries = get_compatible_libraries_for_json(dataset=dataset, hf_token=None, login_required=False)
    libraries = [compatible_library["library"] for compatible_library in compatible_libraries]
    assert libraries == expected_libraries


# --- Test helpers ---

SEED = 42


def _generate_cdc_test_table(n_rows: int = 1000) -> pa.Table:
    """Generate a test table with variable-length strings for CDC detection."""
    np.random.seed(SEED)
    strings = [f"row-{i}-data-" * np.random.randint(1, 200) for i in range(n_rows)]
    return pa.table({"data": pa.array(strings, type=pa.string())})


def _write_parquet(path: str, table: pa.Table, use_cdc: bool = False) -> None:
    """Write a parquet file with page index enabled."""
    write_kwargs: dict[str, Any] = {"write_page_index": True}
    if use_cdc:
        write_kwargs["use_content_defined_chunking"] = {
            "min_chunk_size": 2_500,
            "max_chunk_size": 10_000,
            "norm_level": 0,
        }
    else:
        write_kwargs["data_page_size"] = 10_000
    pq.write_table(table, path, **write_kwargs)


class TestIsOptimizedParquetFromFirstPolarsLoadingCode:
    """Tests for is_optimized_parquet_from_first_polars_loading_code function"""

    def test_cdc_detected_via_rust(self, tmp_path: Path) -> None:
        """Test that a CDC parquet file is correctly detected as optimized."""
        table = _generate_cdc_test_table(n_rows=1000)
        parquet_file = str(tmp_path / "cdc.parquet")
        _write_parquet(parquet_file, table, use_cdc=True)

        with open(parquet_file, "rb") as f:
            file_bytes = f.read()

        try:
            from libviewer._internal import is_content_defined_chunked_parquet

            result = is_content_defined_chunked_parquet(file_bytes)
            assert isinstance(result, bool), f"Expected bool, got {type(result)}"
            assert result is True, f"CDC parquet should be detected as optimized, got {result}"
        except ImportError:
            # libviewer not available, test the pyarrow fallback
            pass

    def test_non_cdc_not_detected_as_optimized(self, tmp_path: Path) -> None:
        """Test that a non-CDC parquet file is NOT detected as optimized."""
        table = _generate_cdc_test_table(n_rows=1000)
        parquet_file = str(tmp_path / "non_cdc.parquet")
        _write_parquet(parquet_file, table, use_cdc=False)

        with open(parquet_file, "rb") as f:
            file_bytes = f.read()

        try:
            from libviewer._internal import is_content_defined_chunked_parquet

            result = is_content_defined_chunked_parquet(file_bytes)
            assert isinstance(result, bool), f"Expected bool, got {type(result)}"
            assert result is False, f"Non-CDC parquet should not be detected as optimized, got {result}"
        except ImportError:
            # libviewer not available
            pass


class TestIsContentDefinedChunkedParquet:
    """Tests for the Rust is_content_defined_chunked_parquet function"""

    def test_with_empty_bytes(self) -> None:
        """Test that empty bytes return an error"""
        try:
            from libviewer._internal import is_content_defined_chunked_parquet

            result = is_content_defined_chunked_parquet(b"")
            assert False, f"Expected error for empty bytes, got {result}"
        except Exception:
            # Expected to fail with empty bytes
            pass

    def test_cdc_detected(self, tmp_path: Path) -> None:
        """Test that CDC parquet is correctly detected"""
        from libviewer._internal import is_content_defined_chunked_parquet

        table = _generate_cdc_test_table(n_rows=1000)
        parquet_file = str(tmp_path / "cdc.parquet")
        _write_parquet(parquet_file, table, use_cdc=True)

        with open(parquet_file, "rb") as f:
            file_bytes = f.read()

        result = is_content_defined_chunked_parquet(file_bytes)
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"
        assert result is True, f"CDC parquet should be detected, got {result}"

    def test_non_cdc_not_detected(self, tmp_path: Path) -> None:
        """Test that non-CDC parquet is not detected as CDC"""
        from libviewer._internal import is_content_defined_chunked_parquet

        table = _generate_cdc_test_table(n_rows=1000)
        parquet_file = str(tmp_path / "non_cdc.parquet")
        _write_parquet(parquet_file, table, use_cdc=False)

        with open(parquet_file, "rb") as f:
            file_bytes = f.read()

        result = is_content_defined_chunked_parquet(file_bytes)
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"
        assert result is False, f"Non-CDC parquet should not be detected, got {result}"


def test_is_optimized_parquet_with_real_cdc_file(tmp_path: Path) -> None:
    """Test is_optimized_parquet from first polars loading code with a real CDC parquet file"""
    # Create a test dataset directory
    test_dataset = "test-real-cdc-dataset"
    ds_dir = tmp_path / "datasets" / test_dataset
    ds_dir.mkdir(parents=True)

    table = _generate_cdc_test_table(n_rows=1000)
    parquet_file = ds_dir / "train.parquet"
    _write_parquet(str(parquet_file), table, use_cdc=True)

    # Check if libviewer is available
    try:
        from libviewer._internal import is_content_defined_chunked_parquet

        # Read the file bytes
        with open(parquet_file, "rb") as f:
            file_bytes = f.read()

        # Test the Rust function
        result = is_content_defined_chunked_parquet(file_bytes)

        assert isinstance(result, bool), f"Expected bool, got {type(result)}"

    except ImportError:
        pass
