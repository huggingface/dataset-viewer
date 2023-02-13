# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

# Adapted from https://github.com/huggingface/datasets/blob/main/tests/fixtures/hub.py

import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Iterable, List, Literal, Mapping, Optional, Tuple, TypedDict

import pytest
import requests
from datasets import Dataset
from huggingface_hub.hf_api import (
    REPO_TYPES,
    REPO_TYPES_URL_PREFIXES,
    HfApi,
    hf_raise_for_status,
)

from ..constants import CI_HUB_ENDPOINT, CI_URL_TEMPLATE, CI_USER, CI_USER_TOKEN

DATASET = "dataset"
hf_api = HfApi(endpoint=CI_HUB_ENDPOINT)


def get_default_config_split(dataset: str) -> Tuple[str, str, str]:
    config = dataset.replace("/", "--")
    split = "train"
    return dataset, config, split


def update_repo_settings(
    *,
    repo_id: str,
    private: Optional[bool] = None,
    gated: Optional[bool] = None,
    token: Optional[str] = None,
    organization: Optional[str] = None,
    repo_type: Optional[str] = None,
    name: str = None,
) -> Mapping[str, bool]:
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
        gated (`bool`, *optional*, defaults to `None`):
            Whether the repo should request user access.
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

    json = {}
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
    *, prefix: str, file_paths: List[str] = None, dataset: Dataset = None, private=False, gated=False
) -> str:
    repo_id = f"{CI_USER}/{prefix}-{int(time.time() * 10e3)}"
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
                path_in_repo=Path(file_path).name,
                repo_id=repo_id,
                repo_type=DATASET,
            )
    return repo_id


def delete_hub_dataset_repo(repo_id: str) -> None:
    with suppress(requests.exceptions.HTTPError, ValueError):
        hf_api.delete_repo(repo_id=repo_id, token=CI_USER_TOKEN, repo_type=DATASET)


# TODO: factor all the datasets fixture with one function that manages the yield and deletion


# https://docs.pytest.org/en/6.2.x/fixture.html#yield-fixtures-recommended
@pytest.fixture(scope="session")
def hub_public_empty() -> Iterable[str]:
    repo_id = create_hub_dataset_repo(prefix="empty")
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_csv(csv_path: str) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(prefix="csv", file_paths=[csv_path])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_private_csv(csv_path: str) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(prefix="csv_private", file_paths=[csv_path], private=True)
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_gated_csv(csv_path: str) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(prefix="csv_gated", file_paths=[csv_path], gated=True)
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_jsonl(jsonl_path: str) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(prefix="jsonl", file_paths=[jsonl_path])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_gated_extra_fields_csv(csv_path: str, extra_fields_readme: str) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(
        prefix="csv_extra_fields_gated", file_paths=[csv_path, extra_fields_readme], gated=True
    )
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_audio(datasets: Mapping[str, Dataset]) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(prefix="audio", dataset=datasets["audio"])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_image(datasets: Mapping[str, Dataset]) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(prefix="image", dataset=datasets["image"])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_images_list(datasets: Mapping[str, Dataset]) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(prefix="images_list", dataset=datasets["images_list"])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


@pytest.fixture(scope="session")
def hub_public_big(datasets: Mapping[str, Dataset]) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(prefix="big", dataset=datasets["big"])
    yield repo_id
    delete_hub_dataset_repo(repo_id=repo_id)


class HubDatasetTest(TypedDict):
    name: str
    config_names_response: Any
    split_names_response: Any
    splits_response: Any
    first_rows_response: Any
    parquet_and_dataset_info_response: Any


HubDatasets = Mapping[str, HubDatasetTest]


def create_config_names_response(dataset: str):
    dataset, config, _ = get_default_config_split(dataset)
    return {
        "config_names": [
            {
                "dataset": dataset,
                "config": config,
            }
        ]
    }


def create_split_names_response(dataset: str):
    dataset, config, split = get_default_config_split(dataset)
    return {
        "split_names": [
            {
                "dataset": dataset,
                "config": config,
                "split": split,
            }
        ]
    }


def create_splits_response(dataset: str):
    dataset, config, split = get_default_config_split(dataset)
    return {
        "splits": [
            {
                "dataset": dataset,
                "config": config,
                "split": split,
            }
        ]
    }


def create_first_rows_response(dataset: str, cols: Mapping[str, Any], rows: List[Any]):
    dataset, config, split = get_default_config_split(dataset)
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
    }


def create_dataset_info_response_for_csv(dataset: str, config: str):
    return {
        "description": "",
        "citation": "",
        "homepage": "",
        "license": "",
        "features": DATA_cols,
        "builder_name": "csv",
        "config_name": config,
        "version": {"version_str": "0.0.0", "major": 0, "minor": 0, "patch": 0},
        "splits": {"train": {"name": "train", "num_bytes": 96, "num_examples": 4, "dataset_name": "csv"}},
        "download_checksums": {
            f"https://hub-ci.huggingface.co/datasets/{dataset}/resolve/__COMMIT__/dataset.csv": {
                "num_bytes": 50,
                "checksum": "441b6927a5442803821415bdcb0f418731b0d2a525a7f2e68ce0df0e95d444de",
            }
        },
        "download_size": 50,
        "dataset_size": 96,
        "size_in_bytes": 146,
    }


def create_dataset_info_response_for_audio(dataset: str, config: str):
    return {
        "description": "",
        "citation": "",
        "homepage": "",
        "license": "",
        "features": AUDIO_cols,
        "splits": {"train": {"name": "train", "num_bytes": 59, "num_examples": 1, "dataset_name": "parquet"}},
        "download_checksums": {
            "SOME_KEY": {
                "num_bytes": 1124,
                "checksum": "3b630ef6ede66c5ced336df78fd99d98f835b459baadbe88a2cdf180709e9543",
            }
        },
        "download_size": 1124,
        "dataset_size": 59,
        "size_in_bytes": 1183,
    }


def create_parquet_and_dataset_info_response(dataset: str, data_type: Literal["csv", "audio"]):
    dataset, config, split = get_default_config_split(dataset)

    filename = "csv-train.parquet" if data_type == "csv" else "parquet-train.parquet"
    size = CSV_PARQUET_SIZE if data_type == "csv" else AUDIO_PARQUET_SIZE
    info = (
        create_dataset_info_response_for_csv(dataset, config)
        if data_type == "csv"
        else create_dataset_info_response_for_audio(dataset, config)
    )
    return {
        "parquet_files": [
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "url": CI_URL_TEMPLATE.format(
                    repo_id=f"datasets/{dataset}", revision="refs%2Fconvert%2Fparquet", filename=f"{config}/{filename}"
                ),
                "filename": filename,
                "size": size,
            }
        ],
        "dataset_info": {config: info},
    }


CSV_PARQUET_SIZE = 1_865
AUDIO_PARQUET_SIZE = 1_383

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


def get_AUDIO_rows(dataset: str):
    dataset, config, split = get_default_config_split(dataset)
    return [
        {
            "col": [
                {
                    "src": f"http://localhost/assets/{dataset}/--/{config}/{split}/0/col/audio.mp3",
                    "type": "audio/mpeg",
                },
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


def get_IMAGE_rows(dataset: str):
    dataset, config, split = get_default_config_split(dataset)
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


def get_IMAGES_LIST_rows(dataset: str):
    dataset, config, split = get_default_config_split(dataset)
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
    "col": [{"_type": "Value", "dtype": "string"}],
}

BIG_rows = ["a" * 1_234 for _ in range(4_567)]


@pytest.fixture(scope="session")
def hub_datasets(
    hub_public_empty,
    hub_public_csv,
    hub_private_csv,
    hub_gated_csv,
    hub_public_jsonl,
    hub_gated_extra_fields_csv,
    hub_public_audio,
    hub_public_image,
    hub_public_images_list,
    hub_public_big,
) -> HubDatasets:
    return {
        "does_not_exist": {
            "name": "does_not_exist",
            "config_names_response": None,
            "split_names_response": None,
            "splits_response": None,
            "first_rows_response": None,
            "parquet_and_dataset_info_response": None,
        },
        "empty": {
            "name": hub_public_empty,
            "config_names_response": None,
            "split_names_response": None,
            "splits_response": None,
            "first_rows_response": None,
            "parquet_and_dataset_info_response": None,
        },
        "public": {
            "name": hub_public_csv,
            "config_names_response": create_config_names_response(hub_public_csv),
            "split_names_response": create_split_names_response(hub_public_csv),
            "splits_response": create_splits_response(hub_public_csv),
            "first_rows_response": create_first_rows_response(hub_public_csv, DATA_cols, DATA_rows),
            "parquet_and_dataset_info_response": create_parquet_and_dataset_info_response(
                dataset=hub_public_csv, data_type="csv"
            ),
        },
        "private": {
            "name": hub_private_csv,
            "config_names_response": create_config_names_response(hub_private_csv),
            "split_names_response": create_split_names_response(hub_private_csv),
            "splits_response": create_splits_response(hub_private_csv),
            "first_rows_response": create_first_rows_response(hub_private_csv, DATA_cols, DATA_rows),
            "parquet_and_dataset_info_response": create_parquet_and_dataset_info_response(
                dataset=hub_private_csv, data_type="csv"
            ),
        },
        "gated": {
            "name": hub_gated_csv,
            "config_names_response": create_config_names_response(hub_gated_csv),
            "split_names_response": create_split_names_response(hub_gated_csv),
            "splits_response": create_splits_response(hub_gated_csv),
            "first_rows_response": create_first_rows_response(hub_gated_csv, DATA_cols, DATA_rows),
            "parquet_and_dataset_info_response": create_parquet_and_dataset_info_response(
                dataset=hub_gated_csv, data_type="csv"
            ),
        },
        "jsonl": {
            "name": hub_public_jsonl,
            "config_names_response": create_config_names_response(hub_public_jsonl),
            "split_names_response": create_split_names_response(hub_public_jsonl),
            "splits_response": create_splits_response(hub_public_jsonl),
            "first_rows_response": create_first_rows_response(hub_public_jsonl, JSONL_cols, JSONL_rows),
            "parquet_and_dataset_info_response": None,
        },
        "gated_extra_fields": {
            "name": hub_gated_extra_fields_csv,
            "config_names_response": create_config_names_response(hub_gated_extra_fields_csv),
            "split_names_response": create_split_names_response(hub_gated_extra_fields_csv),
            "splits_response": create_splits_response(hub_gated_extra_fields_csv),
            "first_rows_response": create_first_rows_response(hub_gated_extra_fields_csv, DATA_cols, DATA_rows),
            "parquet_and_dataset_info_response": create_parquet_and_dataset_info_response(
                dataset=hub_gated_extra_fields_csv, data_type="csv"
            ),
        },
        "audio": {
            "name": hub_public_audio,
            "config_names_response": create_config_names_response(hub_public_audio),
            "split_names_response": create_split_names_response(hub_public_audio),
            "splits_response": create_splits_response(hub_public_audio),
            "first_rows_response": create_first_rows_response(
                hub_public_audio, AUDIO_cols, get_AUDIO_rows(hub_public_audio)
            ),
            "parquet_and_dataset_info_response": create_parquet_and_dataset_info_response(
                dataset=hub_public_audio, data_type="audio"
            ),
        },
        "image": {
            "name": hub_public_image,
            "config_names_response": create_config_names_response(hub_public_image),
            "split_names_response": create_split_names_response(hub_public_image),
            "splits_response": create_splits_response(hub_public_image),
            "first_rows_response": create_first_rows_response(
                hub_public_image, IMAGE_cols, get_IMAGE_rows(hub_public_image)
            ),
            "parquet_and_dataset_info_response": None,
        },
        "images_list": {
            "name": hub_public_images_list,
            "config_names_response": create_config_names_response(hub_public_images_list),
            "split_names_response": create_split_names_response(hub_public_images_list),
            "splits_response": create_splits_response(hub_public_images_list),
            "first_rows_response": create_first_rows_response(
                hub_public_images_list, IMAGES_LIST_cols, get_IMAGES_LIST_rows(hub_public_images_list)
            ),
            "parquet_and_dataset_info_response": None,
        },
        "big": {
            "name": hub_public_big,
            "config_names_response": create_config_names_response(hub_public_big),
            "split_names_response": create_split_names_response(hub_public_big),
            "splits_response": create_splits_response(hub_public_big),
            "first_rows_response": create_first_rows_response(hub_public_big, BIG_cols, BIG_rows),
            "parquet_and_dataset_info_response": None,
        },
    }
