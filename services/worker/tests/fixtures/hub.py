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
from datasets import Dataset, DatasetBuilder, load_dataset_builder
from huggingface_hub.constants import REPO_TYPES, REPO_TYPES_URL_PREFIXES
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils._errors import hf_raise_for_status

from ..constants import CI_HUB_ENDPOINT, CI_URL_TEMPLATE, CI_USER, CI_USER_TOKEN

DATASET = "dataset"
hf_api = HfApi(endpoint=CI_HUB_ENDPOINT)


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
        private (`bool`, *optional*, defaults to `None`):
            Whether the repo should be private.
        gated (`str`, *optional*, defaults to `None`):
            Whether the repo should request user access.
            Possible values are 'auto' and 'manual'
        token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
        repo_type (`str`, *optional*):
            Set to `"dataset"` or `"space"` if uploading to a dataset or
            space, `None` or `"model"` if uploading to a model. Default is
            `None`.
    Returns:
        The HTTP response in json.
    <Tip>
    Raises the following errors:
        - [`~huggingface_hub.utils.RepositoryNotFoundError`]
            If the repository to download from cannot be found. This may be because it doesn't exist,
            or because it is set to `private` and you do not have access.
    </Tip>
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
) -> str:
    dataset_name = f"{prefix}-{int(time.time() * 10e3)}"
    repo_id = f"{CI_USER}/{dataset_name}"
    if dataset is not None:
        dataset.push_to_hub(repo_id=repo_id, private=private, token=CI_USER_TOKEN, embed_external_files=True)
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
    return load_dataset_builder(hub_public_external_files)


@pytest.fixture(scope="session")
def hub_public_legacy_configs(dataset_script_with_two_configs_path: str) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="legacy_configs", file_paths=[dataset_script_with_two_configs_path])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_n_configs(dataset_script_with_n_configs_path: str) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="n_configs", file_paths=[dataset_script_with_n_configs_path])
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
def hub_public_duckdb_index(datasets: Mapping[str, Dataset]) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="duckdb_index", dataset=datasets["duckdb_index"])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_descriptive_statistics(datasets: Mapping[str, Dataset]) -> Iterator[str]:
    repo_id = create_hub_dataset_repo(prefix="descriptive_statistics", dataset=datasets["descriptive_statistics"])
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
    config, _ = get_default_config_split()
    return {
        "config_names": [
            {
                "dataset": dataset,
                "config": config,
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
                "num_bytes": 50,
                "checksum": None,
            }
        },
        "download_size": 50,
        "dataset_size": 96,
        "size_in_bytes": 146,
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
        "splits": {"train": {"name": "train", "num_bytes": 12380, "num_examples": 10, "dataset_name": "csv"}},
        "dataset_size": 12380,
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


def create_dataset_info_response_for_big_parquet_no_info() -> Any:
    return {
        "description": "",
        "citation": "",
        "homepage": "",
        "license": "",
        "features": BIG_cols,
        "splits": {
            "train": {"name": "train", "num_bytes": 12345, "num_examples": len(BIG_rows), "dataset_name": None}
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


def create_parquet_and_info_response(
    dataset: str,
    data_type: Literal["csv", "big-csv", "audio", "big_parquet", "big_parquet_no_info"],
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
        else create_dataset_info_response_for_big_parquet_no_info()
    )
    partial_prefix = "partial-" if partial else ""
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


def get_AUDIO_rows(dataset: str) -> Any:
    config, split = get_default_config_split()
    return [
        {
            "col": [
                {
                    "src": f"http://localhost/assets/{dataset}/--/{config}/{split}/0/col/audio.wav",
                    "type": "audio/wav",
                },
            ]
        }
    ]


IMAGE_cols = {
    "col": {"_type": "Image"},
}


def get_IMAGE_rows(dataset: str) -> Any:
    config, split = get_default_config_split()
    return [
        {
            "col": {
                "src": f"http://localhost/assets/{dataset}/--/{config}/{split}/0/col/image.jpg",
                "height": 480,
                "width": 640,
            },
        }
    ]


IMAGES_LIST_cols = {
    "col": [{"_type": "Image"}],
}


def get_IMAGES_LIST_rows(dataset: str) -> Any:
    config, split = get_default_config_split()
    return [
        {
            "col": [
                {
                    "src": f"http://localhost/assets/{dataset}/--/{config}/{split}/0/col/image-1d100e9.jpg",
                    "height": 480,
                    "width": 640,
                },
                {
                    "src": f"http://localhost/assets/{dataset}/--/{config}/{split}/0/col/image-1d300ea.jpg",
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
            hub_public_audio, AUDIO_cols, get_AUDIO_rows(hub_public_audio)
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
            hub_public_image, IMAGE_cols, get_IMAGE_rows(hub_public_image)
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
            hub_public_images_list, IMAGES_LIST_cols, get_IMAGES_LIST_rows(hub_public_images_list)
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
