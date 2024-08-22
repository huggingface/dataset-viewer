# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

# Adapted from https://github.com/huggingface/datasets/blob/main/tests/fixtures/hub.py

import csv
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict, Union

import pytest
import requests
from datasets import Dataset, DatasetBuilder, Features, Value, load_dataset_builder
from huggingface_hub.constants import REPO_TYPES, REPO_TYPES_URL_PREFIXES
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import hf_raise_for_status
from libcommon.viewer_utils.asset import DATASET_GIT_REVISION_PLACEHOLDER

from ..constants import ASSETS_BASE_URL, CI_HUB_ENDPOINT, CI_URL_TEMPLATE, CI_USER, CI_USER_TOKEN

DATASET = "dataset"
hf_api = HfApi(endpoint=CI_HUB_ENDPOINT)

ANOTHER_CONFIG_NAME = "another_config_name"  # starts with "a" to be able to test ordre wrt "default" config


def get_default_config_split() -> tuple[str, str]:
    config = "default"
    split = "train"
    return config, split


def update_repo_settings(
    *,
    repo_id: str,
    private: Optional[bool] = None,
    gated: Optional[str] = None,
    token: Optional[str] = None,
    organization: Optional[str] = None,
    repo_type: Optional[str] = None,
    name: Optional[str] = None,
) -> Any:
    """Update the settings of a repository.

    Args:
        repo_id (`str`, *optional*):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
            <Tip>
            Version added: 0.5
            </Tip>
        private (`bool`, *optional*):
            Whether the repo should be private.
        gated (`str`, *optional*):
            Whether the repo should request user access.
            Possible values are 'auto' and 'manual'
        token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
        repo_type (`str`, *optional*):
            Set to `"dataset"` or `"space"` if uploading to a dataset or
            space, `None` or `"model"` if uploading to a model.

    Raises:
        [~`huggingface_hub.utils.RepositoryNotFoundError`]:
            If the repository to download from cannot be found. This may be because it doesn't exist,
            or because it is set to `private` and you do not have access.

    Returns:
        `Any`: The HTTP response in json.
    """
    if repo_type not in REPO_TYPES:
        raise ValueError("Invalid repo type")

    organization, name = repo_id.split("/") if "/" in repo_id else (None, repo_id)

    if organization is None:
        namespace = hf_api.whoami(token)["name"]
    else:
        namespace = organization

    path_prefix = f"{hf_api.endpoint}/api/"
    if repo_type in REPO_TYPES_URL_PREFIXES:
        path_prefix += REPO_TYPES_URL_PREFIXES[repo_type]

    path = f"{path_prefix}{namespace}/{name}/settings"

    json: dict[str, Union[bool, str]] = {}
    if private is not None:
        json["private"] = private
    if gated is not None:
        json["gated"] = gated

    r = requests.put(
        path,
        headers={"authorization": f"Bearer {token}"},
        json=json,
    )
    hf_raise_for_status(r)
    return r.json()


def create_hub_dataset_repo(
    *,
    prefix: str,
    file_paths: Optional[list[str]] = None,
    dataset: Optional[Dataset] = None,
    private: bool = False,
    gated: Optional[str] = None,
    configs: Optional[dict[str, Dataset]] = None,
) -> str:
    dataset_name = f"{prefix}-{int(time.time() * 10e3)}"
    repo_id = f"{CI_USER}/{dataset_name}"
    if dataset is not None:
        dataset.push_to_hub(repo_id=repo_id, private=private, token=CI_USER_TOKEN, embed_external_files=True)
    elif configs is not None:
        for config_name, dataset in configs.items():
            dataset.push_to_hub(
                repo_id=repo_id,
                private=private,
                token=CI_USER_TOKEN,
                embed_external_files=True,
                config_name=config_name,
            )
    else:
        hf_api.create_repo(repo_id=repo_id, token=CI_USER_TOKEN, repo_type=DATASET, private=private)
    if gated:
        update_repo_settings(repo_id=repo_id, token=CI_USER_TOKEN, gated=gated, repo_type=DATASET)
    if file_paths is not None:
        for file_path in file_paths:
            hf_api.upload_file(
                token=CI_USER_TOKEN,
                path_or_fileobj=file_path,
                path_in_repo=Path(file_path).name.replace("{dataset_name}", dataset_name),
                repo_id=repo_id,
                repo_type=DATASET,
            )
    return repo_id


def delete_hub_dataset_repo(repo_id: str) -> None:
    with suppress(requests.exceptions.HTTPError, ValueError):
        hf_api.delete_repo(repo_id=repo_id, token=CI_USER_TOKEN, repo_type=DATASET)


# TODO: factor all the datasets fixture with one function that manages the yield and deletion


@pytest.fixture
def tmp_dataset_repo_factory() -> Iterator[Callable[[str], str]]:
    repo_ids: list[str] = []

    def _tmp_dataset_repo(repo_id: str) -> str:
        nonlocal repo_ids
        hf_api.create_repo(repo_id=repo_id, token=CI_USER_TOKEN, repo_type=DATASET)
        repo_ids.append(repo_id)
        return repo_id

    yield _tmp_dataset_repo
    for repo_id in repo_ids:
        delete_hub_dataset_repo(repo_id=repo_id)


# https://docs.pytest.org/en/6.2.x/fixture.html#yield-fixtures-recommended
@pytest.fixture(scope="session")
def hub_public_empty() -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="empty")
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_csv(csv_path: str) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="csv", file_paths=[csv_path])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_private_csv(csv_path: str) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="csv_private", file_paths=[csv_path], private=True)
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_gated_csv(csv_path: str) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="csv_gated", file_paths=[csv_path], gated="auto")
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_gated_duckdb_index(datasets: Mapping[str, Dataset]) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="duckdb_index_gated", dataset=datasets["duckdb_index"], gated="auto")
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_gated_descriptive_statistics(datasets: Mapping[str, Dataset]) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(
        prefix="descriptive_statistics_gated",
        dataset=datasets["descriptive_statistics"],
        gated="auto",
    )
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_jsonl(jsonl_path: str) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="jsonl", file_paths=[jsonl_path])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_audio(datasets: Mapping[str, Dataset]) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="audio", dataset=datasets["audio"])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_image(datasets: Mapping[str, Dataset]) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="image", dataset=datasets["image"])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_images_list(datasets: Mapping[str, Dataset]) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="images_list", dataset=datasets["images_list"])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_big(datasets: Mapping[str, Dataset]) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="big", dataset=datasets["big"])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_big_no_info(datasets: Mapping[str, Dataset]) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="big-no-info", dataset=datasets["big"])
    hf_api.delete_file(
        "README.md", repo_id=repo_id, repo_type="dataset", commit_message="Delete README.md", token=CI_USER_TOKEN
    )
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_big_csv(big_csv_path: str) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="big-csv", file_paths=[big_csv_path])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_external_files(dataset_script_with_external_files_path: str) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="external_files", file_paths=[dataset_script_with_external_files_path])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture
def external_files_dataset_builder(hub_public_external_files: str) -> DatasetBuilder:
    return load_dataset_builder(hub_public_external_files, trust_remote_code=True)


@pytest.fixture(scope="session")
def hub_public_legacy_configs(dataset_script_with_two_configs_path: str) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="legacy_configs", file_paths=[dataset_script_with_two_configs_path])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_legacy_n_configs(dataset_script_with_n_configs_path: str) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="legacy_n_configs", file_paths=[dataset_script_with_n_configs_path])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_manual_download(dataset_script_with_manual_download_path: str) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="manual_download", file_paths=[dataset_script_with_manual_download_path])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_spawning_opt_in_out(datasets: Mapping[str, Dataset]) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="spawning_opt_in_out", dataset=datasets["spawning_opt_in_out"])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_presidio_scan(datasets: Mapping[str, Dataset]) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="presidio_scan", dataset=datasets["presidio_scan"])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_duckdb_index(datasets: Mapping[str, Dataset]) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="duckdb_index", dataset=datasets["duckdb_index"])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_descriptive_statistics(datasets: Mapping[str, Dataset]) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="descriptive_statistics", dataset=datasets["descriptive_statistics"])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_descriptive_statistics_string_text(datasets: Mapping[str, Dataset]) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(
        prefix="descriptive_statistics_string_text", dataset=datasets["descriptive_statistics_string_text"]
    )
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_descriptive_statistics_not_supported(datasets: Mapping[str, Dataset]) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(
        prefix="descriptive_statistics_not_supported", dataset=datasets["descriptive_statistics_not_supported"]
    )
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_audio_statistics(datasets: Mapping[str, Dataset]) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="audio_statistics", dataset=datasets["audio_statistics"])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_image_statistics(datasets: Mapping[str, Dataset]) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="image_statistics", dataset=datasets["image_statistics"])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_n_configs_with_default(datasets: Mapping[str, Dataset]) -> Iterator[str]:
    default_config_name, _ = get_default_config_split()
    repo_id = create_hub_dataset_repo(
        prefix="n_configs",
        configs={
            default_config_name: datasets["descriptive_statistics"],
            ANOTHER_CONFIG_NAME: datasets["descriptive_statistics_string_text"],
        },
    )
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def two_parquet_files_paths(tmp_path_factory: pytest.TempPathFactory, datasets: Mapping[str, Dataset]) -> list[str]:
    # split "descriptive_statistics_string_text" dataset into two parquet files
    dataset = datasets["descriptive_statistics_string_text"]
    path1 = str(tmp_path_factory.mktemp("data") / "0.parquet")
    data1 = Dataset.from_dict(dataset[:50])  # first file - first 50 samples
    with open(path1, "wb") as f:
        data1.to_parquet(f)

    path2 = str(tmp_path_factory.mktemp("data") / "1.parquet")
    data2 = Dataset.from_dict(dataset[50:])  # second file - all the rest
    with open(path2, "wb") as f:
        data2.to_parquet(f)

    return [path1, path2]


@pytest.fixture(scope="session")
def hub_public_descriptive_statistics_parquet_builder(two_parquet_files_paths: list[str]) -> Iterator[str]:
    # to test partial stats, pushing "descriptive_statistics_string_text" dataset split into two parquet files
    # stats will be computed only on the first file (first 50 samples)
    repo_id = create_hub_dataset_repo(prefix="parquet_builder", file_paths=two_parquet_files_paths)
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def three_parquet_files_paths(tmp_path_factory: pytest.TempPathFactory, datasets: Mapping[str, Dataset]) -> list[str]:
    dataset = datasets["descriptive_statistics_string_text"]
    path1 = str(tmp_path_factory.mktemp("data") / "0.parquet")
    data1 = Dataset.from_dict(dataset[:30])
    with open(path1, "wb") as f:
        data1.to_parquet(f)

    path2 = str(tmp_path_factory.mktemp("data") / "1.parquet")
    data2 = Dataset.from_dict(dataset[30:60])
    with open(path2, "wb") as f:
        data2.to_parquet(f)

    path3 = str(tmp_path_factory.mktemp("data") / "2.parquet")
    data3 = Dataset.from_dict(dataset[60:])
    with open(path3, "wb") as f:
        data3.to_parquet(f)

    return [path1, path2, path3]


@pytest.fixture(scope="session")
def hub_public_three_parquet_files_builder(three_parquet_files_paths: list[str]) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="parquet_builder_three_files", file_paths=three_parquet_files_paths)
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def three_parquet_splits_paths(
    tmp_path_factory: pytest.TempPathFactory, datasets: Mapping[str, Dataset]
) -> Mapping[str, str]:
    dataset = datasets["descriptive_statistics_string_text"]
    path1 = str(tmp_path_factory.mktemp("data") / "train.parquet")
    data1 = Dataset.from_dict(dataset[:30])
    with open(path1, "wb") as f:
        data1.to_parquet(f)

    path2 = str(tmp_path_factory.mktemp("data") / "test.parquet")
    data2 = Dataset.from_dict(dataset[30:60])
    with open(path2, "wb") as f:
        data2.to_parquet(f)

    path3 = str(tmp_path_factory.mktemp("data") / "validation.parquet")
    data3 = Dataset.from_dict(dataset[60:])
    with open(path3, "wb") as f:
        data3.to_parquet(f)

    return {"train": path1, "test": path2, "validation": path3}


@pytest.fixture(scope="session")
def hub_public_parquet_splits_empty_rows(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    path = str(tmp_path_factory.mktemp("data") / "train.parquet")
    with open(path, "wb") as f:
        Dataset.from_dict(
            {"col": []},
            features=Features(
                {
                    "col": Value("string"),
                }
            ),
        ).to_parquet(f)

    repo_id = create_hub_dataset_repo(prefix="hub_public_parquet_splits_empty_rows", file_paths=[path])

    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_three_parquet_splits_builder(three_parquet_splits_paths: Mapping[str, str]) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(
        prefix="parquet_builder_three_splits", file_paths=list(three_parquet_splits_paths.values())
    )
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


class HubDatasetTest(TypedDict):
    name: str
    config_names_response: Any
    splits_response: Any
    first_rows_response: Any
    parquet_and_info_response: Any


HubDatasets = Mapping[str, HubDatasetTest]


def create_config_names_response(dataset: str) -> Any:
    default_config_name, _ = get_default_config_split()
    if "n_configs" in dataset:
        return {
            "config_names": [
                {
                    "dataset": dataset,
                    "config": default_config_name,  # default config first
                },
                {
                    "dataset": dataset,
                    "config": ANOTHER_CONFIG_NAME,
                },
            ]
        }
    else:
        return {
            "config_names": [
                {
                    "dataset": dataset,
                    "config": default_config_name,
                }
            ]
        }


def create_splits_response(dataset: str) -> Any:
    config, split = get_default_config_split()
    return {
        "splits": [
            {
                "dataset": dataset,
                "config": config,
                "split": split,
            }
        ]
    }


def create_first_rows_response(dataset: str, cols: Mapping[str, Any], rows: list[Any]) -> Any:
    config, split = get_default_config_split()
    return {
        "dataset": dataset,
        "config": config,
        "split": split,
        "features": [
            {
                "feature_idx": feature_idx,
                "name": name,
                "type": type,
            }
            for feature_idx, (name, type) in enumerate(cols.items())
        ],
        "rows": [
            {
                "row_idx": row_idx,
                "truncated_cells": [],
                "row": row,
            }
            for row_idx, row in enumerate(rows)
        ],
        "truncated": False,
    }


def create_dataset_info_response_for_csv(dataset: str, config: str) -> Any:
    dataset_name = dataset.split("/")[-1]
    return {
        "description": "",
        "citation": "",
        "homepage": "",
        "license": "",
        "features": DATA_cols,
        "builder_name": "csv",
        "config_name": config,
        "dataset_name": dataset_name,
        "version": {"version_str": "0.0.0", "major": 0, "minor": 0, "patch": 0},
        "splits": {"train": {"name": "train", "num_bytes": 96, "num_examples": 4, "dataset_name": dataset_name}},
        "download_checksums": {
            f"https://hub-ci.huggingface.co/datasets/{dataset}/resolve/__COMMIT__/dataset.csv": {
                "num_bytes": 55,
                "checksum": None,
            }
        },
        "download_size": 55,
        "dataset_size": 96,
        "size_in_bytes": 151,
    }


def create_dataset_info_response_for_partially_generated_big_csv(dataset: str, config: str) -> Any:
    # Dataset is partially converted to parquet: the first 10KB instead of the full 5MB
    # Missing fields:
    # - download_size: not applicable, because the dataset is generated using partially downloaded files
    dataset_name = dataset.split("/")[-1]
    return {
        "description": "",
        "citation": "",
        "homepage": "",
        "license": "",
        "features": BIG_cols,
        "builder_name": "csv",
        "config_name": config,
        "dataset_name": dataset_name,
        "version": {"version_str": "0.0.0", "major": 0, "minor": 0, "patch": 0},
        "splits": {"train": {"name": "train", "num_bytes": 12380, "num_examples": 10, "dataset_name": dataset_name}},
        "dataset_size": 12380,
    }


def create_estimated_dataset_info_response_for_partially_generated_big_csv(dataset: str) -> Any:
    # Dataset is partially converted to parquet: the first 10KB instead of the full 5MB
    # Estimation is made based on the ratio of data read vs full data
    dataset_name = dataset.split("/")[-1]
    return {
        "download_size": 5644817,
        "splits": {"train": {"name": "train", "num_bytes": 266581, "num_examples": 215, "dataset_name": dataset_name}},
        "dataset_size": 266581,
    }


def create_dataset_info_response_for_big_parquet(dataset: str, config: str) -> Any:
    dataset_name = dataset.split("/")[-1]
    return {
        "description": "",
        "citation": "",
        "homepage": "",
        "license": "",
        "features": BIG_cols,
        "builder_name": "parquet",
        "config_name": config,
        "dataset_name": dataset_name,
        "version": {"version_str": "0.0.0", "major": 0, "minor": 0, "patch": 0},
        "splits": {
            "train": {"name": "train", "num_bytes": 5653946, "num_examples": len(BIG_rows), "dataset_name": None}
        },
        "download_size": BIG_PARQUET_FILE,
        "dataset_size": 5653946,
    }


def create_dataset_info_response_for_big_parquet_no_info(dataset: str) -> Any:
    dataset_name = dataset.split("/")[-1]
    return {
        "description": "",
        "citation": "",
        "homepage": "",
        "license": "",
        "features": BIG_cols,
        "splits": {
            "train": {"name": "train", "num_bytes": 12345, "num_examples": len(BIG_rows), "dataset_name": dataset_name}
        },
        "download_size": BIG_PARQUET_FILE,
        "dataset_size": 12345,
    }


def create_dataset_info_response_for_audio(dataset: str, config: str) -> Any:
    dataset_name = dataset.split("/")[-1]
    return {
        "description": "",
        "citation": "",
        "homepage": "",
        "license": "",
        "features": AUDIO_cols,
        "builder_name": "parquet",
        "config_name": config,
        "dataset_name": dataset_name,
        "version": {"version_str": "0.0.0", "major": 0, "minor": 0, "patch": 0},
        "splits": {"train": {"name": "train", "num_bytes": 59, "num_examples": 1, "dataset_name": None}},
        "download_size": AUDIO_PARQUET_SIZE,
        "dataset_size": 59,
    }


def create_dataset_info_response_for_presidio_scan(dataset: str, config: str) -> Any:
    dataset_name = dataset.split("/")[-1]
    return {
        "description": "",
        "citation": "",
        "homepage": "",
        "license": "",
        "features": PRESIDIO_SCAN_cols,
        "builder_name": "parquet",
        "config_name": config,
        "dataset_name": dataset_name,
        "version": {"version_str": "0.0.0", "major": 0, "minor": 0, "patch": 0},
        "splits": {"train": {"name": "train", "num_bytes": 12345, "num_examples": 6, "dataset_name": dataset_name}},
        "download_size": 12345,
        "dataset_size": 12345,
    }


def create_parquet_and_info_response(
    dataset: str,
    data_type: Literal["csv", "big-csv", "audio", "big_parquet", "big_parquet_no_info", "presidio-scan"],
    partial: bool = False,
) -> Any:
    config, split = get_default_config_split()
    filename = "0000.parquet"
    size = (
        CSV_PARQUET_SIZE
        if data_type == "csv"
        else PARTIAL_CSV_PARQUET_SIZE
        if data_type == "big-csv"
        else AUDIO_PARQUET_SIZE
        if data_type == "audio"
        else BIG_PARQUET_FILE
    )
    info = (
        create_dataset_info_response_for_csv(dataset, config)
        if data_type == "csv"
        else create_dataset_info_response_for_partially_generated_big_csv(dataset, config)
        if data_type == "big-csv"
        else create_dataset_info_response_for_audio(dataset, config)
        if data_type == "audio"
        else create_dataset_info_response_for_big_parquet(dataset, config)
        if data_type == "big_parquet"
        else create_dataset_info_response_for_presidio_scan(dataset, config)
        if data_type == "presidio-scan"
        else create_dataset_info_response_for_big_parquet_no_info(dataset)
    )
    partial_prefix = "partial-" if partial else ""
    estimated_info = (
        create_estimated_dataset_info_response_for_partially_generated_big_csv(dataset)
        if data_type == "big-csv"
        else None
    )
    return {
        "parquet_files": [
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "url": CI_URL_TEMPLATE.format(
                    repo_id=f"datasets/{dataset}",
                    revision="refs%2Fconvert%2Fparquet",
                    filename=f"{config}/{partial_prefix}{split}/{filename}",
                ),
                "filename": filename,
                "size": size,
            }
        ],
        "dataset_info": info,
        "estimated_dataset_info": estimated_info,
        "partial": partial,
    }


CSV_PARQUET_SIZE = 1_866
PARTIAL_CSV_PARQUET_SIZE = 8_188
AUDIO_PARQUET_SIZE = 1_384
BIG_PARQUET_FILE = 38_896

DATA_cols = {
    "col_1": {"_type": "Value", "dtype": "int64"},
    "col_2": {"_type": "Value", "dtype": "int64"},
    "col_3": {"_type": "Value", "dtype": "float64"},
}
DATA_rows = [
    {"col_1": 0, "col_2": 0, "col_3": 0.0},
    {"col_1": 1, "col_2": 1, "col_3": 1.0},
    {"col_1": 2, "col_2": 2, "col_3": 2.0},
    {"col_1": 3, "col_2": 3, "col_3": 3.0},
]


JSONL_cols = {
    "col_1": {"_type": "Value", "dtype": "string"},
    "col_2": {"_type": "Value", "dtype": "int64"},
    "col_3": {"_type": "Value", "dtype": "float64"},
}
JSONL_rows = [
    {"col_1": "0", "col_2": 0, "col_3": 0.0},
    {"col_1": None, "col_2": 1, "col_3": 1.0},
    {"col_1": None, "col_2": 2, "col_3": 2.0},
    {"col_1": "3", "col_2": 3, "col_3": 3.0},
]

AUDIO_cols = {
    "col": {
        "_type": "Audio",
        "sampling_rate": 16_000,
    },
}


def get_AUDIO_first_rows_response(dataset: str) -> Any:
    config, split = get_default_config_split()
    return [
        {
            "col": [
                {
                    "src": f"http://localhost/assets/{dataset}/--/{DATASET_GIT_REVISION_PLACEHOLDER}/--/{config}/{split}/0/col/audio.wav",
                    "type": "audio/wav",
                },
            ]
        }
    ]


IMAGE_cols = {
    "col": {"_type": "Image"},
}


def get_IMAGE_first_rows_response(dataset: str) -> Any:
    config, split = get_default_config_split()
    return [
        {
            "col": {
                "src": f"http://localhost/assets/{dataset}/--/{DATASET_GIT_REVISION_PLACEHOLDER}/--/{config}/{split}/0/col/image.jpg",
                "height": 480,
                "width": 640,
            },
        }
    ]


IMAGES_LIST_cols = {
    "col": [{"_type": "Image"}],
}


def get_IMAGES_LIST_first_rows_response(dataset: str) -> Any:
    config, split = get_default_config_split()
    return [
        {
            "col": [
                {
                    "src": (
                        f"{ASSETS_BASE_URL}/{dataset}/--/{DATASET_GIT_REVISION_PLACEHOLDER}/--/{config}/{split}/0/col/image-1d100e9.jpg"
                    ),
                    "height": 480,
                    "width": 640,
                },
                {
                    "src": (
                        f"{ASSETS_BASE_URL}/{dataset}/--/{DATASET_GIT_REVISION_PLACEHOLDER}/--/{config}/{split}/0/col/image-1d300ea.jpg"
                    ),
                    "height": 480,
                    "width": 640,
                },
            ]
        }
    ]


BIG_cols = {
    "col": {"_type": "Value", "dtype": "string"},
}

BIG_rows = [{"col": "a" * 1_234} for _ in range(4_567)]


@pytest.fixture(scope="session")
def big_csv_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "big_dataset.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(BIG_cols))
        writer.writeheader()
        for row in BIG_rows:
            writer.writerow(row)
    return path


TEXT_cols = {
    "text": {"_type": "Value", "dtype": "string"},
}

LARGE_TEXT_cols = {
    "text": {"_type": "Value", "dtype": "large_string"},
}

TEXT_rows = [
    {"text": text}
    for text in [
        "foo",
        "bar",
        "foobar",
        "- Hello there !",
        "- General Kenobi !",
    ]
]


SPAWNING_OPT_IN_OUT_cols = {
    "col": [{"_type": "Value", "dtype": "string"}],
}

SPAWNING_OPT_IN_OUT_rows = ["http://testurl.test/test_image.jpg", "http://testurl.test/test_image2.jpg", "other"]

PRESIDIO_SCAN_cols = {
    "col": [{"_type": "Value", "dtype": "string"}],
}

PRESIDIO_SCAN_rows = [
    {"col": text}
    for text in [
        "My name is Giovanni Giorgio",
        "but everyone calls me Giorgio",
        "My IP address is 192.168.0.1",
        "My SSN is 345-67-8901",
        "My email is giovanni.giorgio@daftpunk.com",
        None,
    ]
]


@pytest.fixture
def hub_responses_does_not_exist() -> HubDatasetTest:
    return {
        "name": "does_not_exist",
        "config_names_response": None,
        "splits_response": None,
        "first_rows_response": None,
        "parquet_and_info_response": None,
    }


@pytest.fixture
def hub_responses_does_not_exist_config() -> HubDatasetTest:
    return {
        "name": "does_not_exist_config",
        "config_names_response": None,
        "splits_response": None,
        "first_rows_response": None,
        "parquet_and_info_response": None,
    }


@pytest.fixture
def hub_responses_does_not_exist_split() -> HubDatasetTest:
    return {
        "name": "does_not_exist_split",
        "config_names_response": None,
        "splits_response": None,
        "first_rows_response": None,
        "parquet_and_info_response": None,
    }


@pytest.fixture
def hub_responses_empty(hub_public_empty: str) -> HubDatasetTest:
    return {
        "name": hub_public_empty,
        "config_names_response": None,
        "splits_response": None,
        "first_rows_response": None,
        "parquet_and_info_response": None,
    }


@pytest.fixture
def hub_responses_public(hub_public_csv: str) -> HubDatasetTest:
    return {
        "name": hub_public_csv,
        "config_names_response": create_config_names_response(hub_public_csv),
        "splits_response": create_splits_response(hub_public_csv),
        "first_rows_response": create_first_rows_response(hub_public_csv, DATA_cols, DATA_rows),
        "parquet_and_info_response": create_parquet_and_info_response(dataset=hub_public_csv, data_type="csv"),
    }


@pytest.fixture
def hub_responses_private(hub_private_csv: str) -> HubDatasetTest:
    return {
        "name": hub_private_csv,
        "config_names_response": create_config_names_response(hub_private_csv),
        "splits_response": create_splits_response(hub_private_csv),
        "first_rows_response": create_first_rows_response(hub_private_csv, DATA_cols, DATA_rows),
        "parquet_and_info_response": create_parquet_and_info_response(dataset=hub_private_csv, data_type="csv"),
    }


@pytest.fixture
def hub_responses_gated(hub_gated_csv: str) -> HubDatasetTest:
    return {
        "name": hub_gated_csv,
        "config_names_response": create_config_names_response(hub_gated_csv),
        "splits_response": create_splits_response(hub_gated_csv),
        "first_rows_response": create_first_rows_response(hub_gated_csv, DATA_cols, DATA_rows),
        "parquet_and_info_response": create_parquet_and_info_response(dataset=hub_gated_csv, data_type="csv"),
    }


@pytest.fixture
def hub_reponses_jsonl(hub_public_jsonl: str) -> HubDatasetTest:
    return {
        "name": hub_public_jsonl,
        "config_names_response": create_config_names_response(hub_public_jsonl),
        "splits_response": create_splits_response(hub_public_jsonl),
        "first_rows_response": create_first_rows_response(hub_public_jsonl, JSONL_cols, JSONL_rows),
        "parquet_and_info_response": None,
    }


@pytest.fixture
def hub_responses_audio(hub_public_audio: str) -> HubDatasetTest:
    return {
        "name": hub_public_audio,
        "config_names_response": create_config_names_response(hub_public_audio),
        "splits_response": create_splits_response(hub_public_audio),
        "first_rows_response": create_first_rows_response(
            hub_public_audio, AUDIO_cols, get_AUDIO_first_rows_response(hub_public_audio)
        ),
        "parquet_and_info_response": create_parquet_and_info_response(dataset=hub_public_audio, data_type="audio"),
    }


@pytest.fixture
def hub_responses_image(hub_public_image: str) -> HubDatasetTest:
    return {
        "name": hub_public_image,
        "config_names_response": create_config_names_response(hub_public_image),
        "splits_response": create_splits_response(hub_public_image),
        "first_rows_response": create_first_rows_response(
            hub_public_image, IMAGE_cols, get_IMAGE_first_rows_response(hub_public_image)
        ),
        "parquet_and_info_response": None,
    }


@pytest.fixture
def hub_responses_images_list(hub_public_images_list: str) -> HubDatasetTest:
    return {
        "name": hub_public_images_list,
        "config_names_response": create_config_names_response(hub_public_images_list),
        "splits_response": create_splits_response(hub_public_images_list),
        "first_rows_response": create_first_rows_response(
            hub_public_images_list, IMAGES_LIST_cols, get_IMAGES_LIST_first_rows_response(hub_public_images_list)
        ),
        "parquet_and_info_response": None,
    }


@pytest.fixture
def hub_responses_big(hub_public_big: str) -> HubDatasetTest:
    return {
        "name": hub_public_big,
        "config_names_response": create_config_names_response(hub_public_big),
        "splits_response": create_splits_response(hub_public_big),
        "first_rows_response": create_first_rows_response(hub_public_big, BIG_cols, BIG_rows),
        "parquet_and_info_response": create_parquet_and_info_response(dataset=hub_public_big, data_type="big_parquet"),
    }


@pytest.fixture
def hub_responses_big_no_info(hub_public_big_no_info: str) -> HubDatasetTest:
    return {
        "name": hub_public_big_no_info,
        "config_names_response": create_config_names_response(hub_public_big_no_info),
        "splits_response": create_splits_response(hub_public_big_no_info),
        "first_rows_response": create_first_rows_response(hub_public_big_no_info, BIG_cols, BIG_rows),
        "parquet_and_info_response": create_parquet_and_info_response(
            dataset=hub_public_big_no_info, data_type="big_parquet_no_info"
        ),
    }


@pytest.fixture
def hub_responses_big_csv(hub_public_big_csv: str) -> HubDatasetTest:
    return {
        "name": hub_public_big_csv,
        "config_names_response": create_config_names_response(hub_public_big_csv),
        "splits_response": create_splits_response(hub_public_big_csv),
        "first_rows_response": create_first_rows_response(hub_public_big_csv, BIG_cols, BIG_rows),
        "parquet_and_info_response": create_parquet_and_info_response(
            dataset=hub_public_big_csv, data_type="big-csv", partial=True
        ),
    }


@pytest.fixture
def hub_responses_external_files(hub_public_external_files: str) -> HubDatasetTest:
    return {
        "name": hub_public_external_files,
        "config_names_response": create_config_names_response(hub_public_external_files),
        "splits_response": create_splits_response(hub_public_external_files),
        "first_rows_response": create_first_rows_response(hub_public_external_files, TEXT_cols, TEXT_rows),
        "parquet_and_info_response": None,
    }


@pytest.fixture
def hub_responses_spawning_opt_in_out(hub_public_spawning_opt_in_out: str) -> HubDatasetTest:
    return {
        "name": hub_public_spawning_opt_in_out,
        "config_names_response": create_config_names_response(hub_public_spawning_opt_in_out),
        "splits_response": create_splits_response(hub_public_spawning_opt_in_out),
        "first_rows_response": create_first_rows_response(
            hub_public_spawning_opt_in_out, SPAWNING_OPT_IN_OUT_cols, SPAWNING_OPT_IN_OUT_rows
        ),
        "parquet_and_info_response": None,
    }


@pytest.fixture
def hub_responses_presidio_scan(hub_public_presidio_scan: str) -> HubDatasetTest:
    return {
        "name": hub_public_presidio_scan,
        "config_names_response": create_config_names_response(hub_public_presidio_scan),
        "splits_response": create_splits_response(hub_public_presidio_scan),
        "first_rows_response": create_first_rows_response(
            hub_public_presidio_scan, PRESIDIO_SCAN_cols, PRESIDIO_SCAN_rows
        ),
        "parquet_and_info_response": create_parquet_and_info_response(
            dataset=hub_public_presidio_scan, data_type="presidio-scan"
        ),
    }


@pytest.fixture
def hub_responses_duckdb_index(hub_public_duckdb_index: str) -> HubDatasetTest:
    return {
        "name": hub_public_duckdb_index,
        "config_names_response": create_config_names_response(hub_public_duckdb_index),
        "splits_response": create_splits_response(hub_public_duckdb_index),
        "first_rows_response": create_first_rows_response(hub_public_duckdb_index, TEXT_cols, TEXT_rows),
        "parquet_and_info_response": create_parquet_and_info_response(
            dataset=hub_public_duckdb_index, data_type="csv"
        ),
    }


@pytest.fixture
def hub_responses_partial_duckdb_index(hub_public_duckdb_index: str) -> HubDatasetTest:
    return {
        "name": hub_public_duckdb_index,
        "config_names_response": create_config_names_response(hub_public_duckdb_index),
        "splits_response": create_splits_response(hub_public_duckdb_index),
        "first_rows_response": create_first_rows_response(hub_public_duckdb_index, TEXT_cols, TEXT_rows),
        "parquet_and_info_response": create_parquet_and_info_response(
            dataset=hub_public_duckdb_index, data_type="csv", partial=True
        ),
    }


@pytest.fixture
def hub_responses_gated_duckdb_index(hub_gated_duckdb_index: str) -> HubDatasetTest:
    return {
        "name": hub_gated_duckdb_index,
        "config_names_response": create_config_names_response(hub_gated_duckdb_index),
        "splits_response": create_splits_response(hub_gated_duckdb_index),
        "first_rows_response": create_first_rows_response(hub_gated_duckdb_index, TEXT_cols, TEXT_rows),
        "parquet_and_info_response": create_parquet_and_info_response(dataset=hub_gated_duckdb_index, data_type="csv"),
    }


@pytest.fixture
def hub_responses_descriptive_statistics(hub_public_descriptive_statistics: str) -> HubDatasetTest:
    return {
        "name": hub_public_descriptive_statistics,
        "config_names_response": create_config_names_response(hub_public_descriptive_statistics),
        "splits_response": create_splits_response(hub_public_descriptive_statistics),
        "first_rows_response": None,
        "parquet_and_info_response": None,
    }


@pytest.fixture
def hub_responses_gated_descriptive_statistics(hub_gated_descriptive_statistics: str) -> HubDatasetTest:
    return {
        "name": hub_gated_descriptive_statistics,
        "config_names_response": create_config_names_response(hub_gated_descriptive_statistics),
        "splits_response": create_splits_response(hub_gated_descriptive_statistics),
        "first_rows_response": None,
        "parquet_and_info_response": None,
    }


@pytest.fixture
def hub_responses_descriptive_statistics_string_text(
    hub_public_descriptive_statistics_string_text: str,
) -> HubDatasetTest:
    return {
        "name": hub_public_descriptive_statistics_string_text,
        "config_names_response": create_config_names_response(hub_public_descriptive_statistics_string_text),
        "splits_response": create_splits_response(hub_public_descriptive_statistics_string_text),
        "first_rows_response": None,
        "parquet_and_info_response": None,
    }


@pytest.fixture
def hub_responses_descriptive_statistics_not_supported(
    hub_public_descriptive_statistics_not_supported: str,
) -> HubDatasetTest:
    return {
        "name": hub_public_descriptive_statistics_not_supported,
        "config_names_response": create_config_names_response(hub_public_descriptive_statistics_not_supported),
        "splits_response": create_splits_response(hub_public_descriptive_statistics_not_supported),
        "first_rows_response": None,
        "parquet_and_info_response": None,
    }


@pytest.fixture
def hub_responses_audio_statistics(
    hub_public_audio_statistics: str,
) -> HubDatasetTest:
    return {
        "name": hub_public_audio_statistics,
        "config_names_response": create_config_names_response(hub_public_audio_statistics),
        "splits_response": create_splits_response(hub_public_audio_statistics),
        "first_rows_response": None,
        "parquet_and_info_response": None,
    }


@pytest.fixture
def hub_responses_image_statistics(
    hub_public_image_statistics: str,
) -> HubDatasetTest:
    return {
        "name": hub_public_image_statistics,
        "config_names_response": create_config_names_response(hub_public_image_statistics),
        "splits_response": create_splits_response(hub_public_image_statistics),
        "first_rows_response": None,
        "parquet_and_info_response": None,
    }


@pytest.fixture
def hub_responses_descriptive_statistics_parquet_builder(
    hub_public_descriptive_statistics_parquet_builder: str,
) -> HubDatasetTest:
    return {
        "name": hub_public_descriptive_statistics_parquet_builder,
        "config_names_response": create_config_names_response(hub_public_descriptive_statistics_parquet_builder),
        "splits_response": create_splits_response(hub_public_descriptive_statistics_parquet_builder),
        "first_rows_response": None,
        "parquet_and_info_response": None,
    }


@pytest.fixture
def hub_responses_n_configs_with_default(
    hub_public_n_configs_with_default: str,
) -> HubDatasetTest:
    return {
        "name": hub_public_n_configs_with_default,
        "config_names_response": create_config_names_response(hub_public_n_configs_with_default),
        "splits_response": None,
        "first_rows_response": None,
        "parquet_and_info_response": None,
    }
