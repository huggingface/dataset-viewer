# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

# Adapted from https://github.com/huggingface/datasets/blob/main/tests/fixtures/hub.py

import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Tuple, TypedDict

import datasets.config
import pytest
import requests
from datasets import Dataset
from huggingface_hub.hf_api import (
    REPO_TYPES,
    REPO_TYPES_URL_PREFIXES,
    HfApi,
    hf_raise_for_status,
)

# see https://github.com/huggingface/moon-landing/blob/main/server/scripts/staging-seed-db.ts
CI_HUB_USER = "__DUMMY_DATASETS_SERVER_USER__"
CI_HUB_USER_API_TOKEN = "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"

CI_HUB_ENDPOINT = "https://hub-ci.huggingface.co"
CI_HFH_HUGGINGFACE_CO_URL_TEMPLATE = CI_HUB_ENDPOINT + "/{repo_id}/resolve/{revision}/{filename}"


@pytest.fixture(autouse=True)
def ci_hfh_hf_hub_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "huggingface_hub.file_download.HUGGINGFACE_CO_URL_TEMPLATE", CI_HFH_HUGGINGFACE_CO_URL_TEMPLATE
    )


# Ensure the datasets library uses the expected HuggingFace endpoint
datasets.config.HF_ENDPOINT = CI_HUB_ENDPOINT
# Don't increase the datasets download counts on huggingface.co
datasets.config.HF_UPDATE_DOWNLOAD_COUNTS = False


def get_default_config_split(dataset: str) -> Tuple[str, str, str]:
    config = dataset.replace("/", "--")
    split = "train"
    return dataset, config, split


def update_repo_settings(
    hf_api: HfApi,
    repo_id: str,
    *,
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


@pytest.fixture(scope="session")
def hf_api():
    return HfApi(endpoint=CI_HUB_ENDPOINT)


@pytest.fixture(scope="session")
def hf_token() -> str:
    return CI_HUB_USER_API_TOKEN


@pytest.fixture(scope="session")
def hf_endpoint() -> str:
    return CI_HUB_ENDPOINT


@pytest.fixture
def cleanup_repo(hf_api: HfApi):
    def _cleanup_repo(repo_id):
        hf_api.delete_repo(repo_id=repo_id, token=CI_HUB_USER_API_TOKEN, repo_type="dataset")

    return _cleanup_repo


@pytest.fixture
def temporary_repo(cleanup_repo):
    @contextmanager
    def _temporary_repo(repo_id):
        try:
            yield repo_id
        finally:
            cleanup_repo(repo_id)

    return _temporary_repo


def create_unique_repo_name(prefix: str, user: str) -> str:
    repo_name = f"{prefix}-{int(time.time() * 10e3)}"
    return f"{user}/{repo_name}"


def create_hub_dataset_repo(
    *,
    hf_api: HfApi,
    hf_token: str,
    prefix: str,
    file_paths: List[str] = None,
    dataset: Dataset = None,
    private=False,
    gated=False,
    user=CI_HUB_USER,
) -> str:
    repo_id = create_unique_repo_name(prefix, user)
    if dataset is not None:
        dataset.push_to_hub(repo_id=repo_id, private=private, token=hf_token, embed_external_files=True)
    else:
        hf_api.create_repo(repo_id=repo_id, token=hf_token, repo_type="dataset", private=private)
    if gated:
        update_repo_settings(hf_api, repo_id, token=hf_token, gated=gated, repo_type="dataset")
    if file_paths is not None:
        for file_path in file_paths:
            hf_api.upload_file(
                token=hf_token,
                path_or_fileobj=file_path,
                path_in_repo=Path(file_path).name,
                repo_id=repo_id,
                repo_type="dataset",
            )
    return repo_id


# https://docs.pytest.org/en/6.2.x/fixture.html#yield-fixtures-recommended
@pytest.fixture(scope="session", autouse=True)
def hub_public_empty(hf_api: HfApi, hf_token: str) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(hf_api=hf_api, hf_token=hf_token, prefix="empty")
    yield repo_id
    with suppress(requests.exceptions.HTTPError, ValueError):
        hf_api.delete_repo(repo_id=repo_id, token=hf_token, repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def hub_public_csv(hf_api: HfApi, hf_token: str, csv_path: str) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(hf_api=hf_api, hf_token=hf_token, prefix="csv", file_paths=[csv_path])
    yield repo_id
    with suppress(requests.exceptions.HTTPError, ValueError):
        hf_api.delete_repo(repo_id=repo_id, token=hf_token, repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def hub_private_csv(hf_api: HfApi, hf_token: str, csv_path: str) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(
        hf_api=hf_api, hf_token=hf_token, prefix="csv_private", file_paths=[csv_path], private=True
    )
    yield repo_id
    with suppress(requests.exceptions.HTTPError, ValueError):
        hf_api.delete_repo(repo_id=repo_id, token=hf_token, repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def hub_gated_csv(hf_api: HfApi, hf_token: str, csv_path: str) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(
        hf_api=hf_api, hf_token=hf_token, prefix="csv_gated", file_paths=[csv_path], gated=True
    )
    yield repo_id
    with suppress(requests.exceptions.HTTPError, ValueError):
        hf_api.delete_repo(repo_id=repo_id, token=hf_token, repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def hub_public_jsonl(hf_api: HfApi, hf_token: str, jsonl_path: str) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(hf_api=hf_api, hf_token=hf_token, prefix="jsonl", file_paths=[jsonl_path])
    yield repo_id
    with suppress(requests.exceptions.HTTPError, ValueError):
        hf_api.delete_repo(repo_id=repo_id, token=hf_token, repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def hub_gated_extra_fields_csv(hf_api: HfApi, hf_token: str, csv_path: str, extra_fields_readme: str) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(
        hf_api=hf_api,
        hf_token=hf_token,
        prefix="csv_extra_fields_gated",
        file_paths=[csv_path, extra_fields_readme],
        gated=True,
    )
    yield repo_id
    with suppress(requests.exceptions.HTTPError, ValueError):
        hf_api.delete_repo(repo_id=repo_id, token=hf_token, repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def hub_public_audio(hf_api: HfApi, hf_token: str, datasets: Mapping[str, Dataset]) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(hf_api=hf_api, hf_token=hf_token, prefix="audio", dataset=datasets["audio"])
    yield repo_id
    with suppress(requests.exceptions.HTTPError, ValueError):
        hf_api.delete_repo(repo_id=repo_id, token=hf_token, repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def hub_public_image(hf_api: HfApi, hf_token: str, datasets: Mapping[str, Dataset]) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(hf_api=hf_api, hf_token=hf_token, prefix="image", dataset=datasets["image"])
    yield repo_id
    with suppress(requests.exceptions.HTTPError, ValueError):
        hf_api.delete_repo(repo_id=repo_id, token=hf_token, repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def hub_public_images_list(hf_api: HfApi, hf_token: str, datasets: Mapping[str, Dataset]) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(
        hf_api=hf_api, hf_token=hf_token, prefix="images_list", dataset=datasets["images_list"]
    )
    yield repo_id
    with suppress(requests.exceptions.HTTPError, ValueError):
        hf_api.delete_repo(repo_id=repo_id, token=hf_token, repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def hub_public_big(hf_api: HfApi, hf_token: str, datasets: Mapping[str, Dataset]) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(hf_api=hf_api, hf_token=hf_token, prefix="big", dataset=datasets["big"])
    yield repo_id
    with suppress(requests.exceptions.HTTPError, ValueError):
        hf_api.delete_repo(repo_id=repo_id, token=hf_token, repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def hub_not_supported_csv(hf_api: HfApi, hf_token: str, csv_path: str) -> Iterable[str]:
    repo_id = create_hub_dataset_repo(hf_api=hf_api, hf_token=hf_token, prefix="not_supported", file_paths=[csv_path])
    yield repo_id
    with suppress(requests.exceptions.HTTPError, ValueError):
        hf_api.delete_repo(repo_id=repo_id, token=hf_token, repo_type="dataset")


class HubDatasetTest(TypedDict):
    name: str
    splits_response: Any
    first_rows_response: Any
    parquet_response: Any


HubDatasets = Mapping[str, HubDatasetTest]


def create_splits_response(dataset: str, num_bytes: float = None, num_examples: int = None):
    dataset, config, split = get_default_config_split(dataset)
    return {
        "splits": [
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "num_bytes": num_bytes,
                "num_examples": num_examples,
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


def create_parquet_response(dataset: str, filename: str, size: int):
    dataset, config, split = get_default_config_split(dataset)
    return {
        "parquet_files": [
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "url": CI_HFH_HUGGINGFACE_CO_URL_TEMPLATE.format(
                    repo_id=f"datasets/{dataset}", revision="refs%2Fconvert%2Fparquet", filename=f"{config}/{filename}"
                ),
                "filename": filename,
                "size": size,
            }
        ],
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


@pytest.fixture(scope="session", autouse=True)
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
            "splits_response": None,
            "first_rows_response": None,
            "parquet_response": None,
        },
        "empty": {
            "name": hub_public_empty,
            "splits_response": None,
            "first_rows_response": None,
            "parquet_response": None,
        },
        "public": {
            "name": hub_public_csv,
            "splits_response": create_splits_response(hub_public_csv, None, None),
            "first_rows_response": create_first_rows_response(hub_public_csv, DATA_cols, DATA_rows),
            "parquet_response": create_parquet_response(
                dataset=hub_public_csv, filename="csv-train.parquet", size=CSV_PARQUET_SIZE
            ),
        },
        "private": {
            "name": hub_private_csv,
            "splits_response": create_splits_response(hub_private_csv, None, None),
            "first_rows_response": create_first_rows_response(hub_private_csv, DATA_cols, DATA_rows),
            "parquet_response": create_parquet_response(
                dataset=hub_private_csv, filename="csv-train.parquet", size=CSV_PARQUET_SIZE
            ),
        },
        "gated": {
            "name": hub_gated_csv,
            "splits_response": create_splits_response(hub_gated_csv, None, None),
            "first_rows_response": create_first_rows_response(hub_gated_csv, DATA_cols, DATA_rows),
            "parquet_response": create_parquet_response(
                dataset=hub_gated_csv, filename="csv-train.parquet", size=CSV_PARQUET_SIZE
            ),
        },
        "jsonl": {
            "name": hub_public_jsonl,
            "splits_response": create_splits_response(hub_public_jsonl, None, None),
            "first_rows_response": create_first_rows_response(hub_public_jsonl, JSONL_cols, JSONL_rows),
            "parquet_response": None,
        },
        "gated_extra_fields": {
            "name": hub_gated_extra_fields_csv,
            "splits_response": create_splits_response(hub_gated_extra_fields_csv, None, None),
            "first_rows_response": create_first_rows_response(hub_gated_extra_fields_csv, DATA_cols, DATA_rows),
            "parquet_response": create_parquet_response(
                dataset=hub_gated_extra_fields_csv, filename="csv-train.parquet", size=CSV_PARQUET_SIZE
            ),
        },
        "audio": {
            "name": hub_public_audio,
            "splits_response": create_splits_response(hub_public_audio, 54.0, 1),
            "first_rows_response": create_first_rows_response(
                hub_public_audio, AUDIO_cols, get_AUDIO_rows(hub_public_audio)
            ),
            "parquet_response": create_parquet_response(
                dataset=hub_public_audio,
                filename="parquet-train.parquet",
                size=AUDIO_PARQUET_SIZE,
            ),
        },
        "image": {
            "name": hub_public_image,
            "splits_response": create_splits_response(hub_public_image, 0, 1),
            "first_rows_response": create_first_rows_response(
                hub_public_image, IMAGE_cols, get_IMAGE_rows(hub_public_image)
            ),
            "parquet_response": None,
        },
        "images_list": {
            "name": hub_public_images_list,
            "splits_response": create_splits_response(hub_public_images_list, 0, 1),
            "first_rows_response": create_first_rows_response(
                hub_public_images_list, IMAGES_LIST_cols, get_IMAGES_LIST_rows(hub_public_images_list)
            ),
            "parquet_response": None,
        },
        "big": {
            "name": hub_public_big,
            "splits_response": create_splits_response(hub_public_big, 0, 1),
            "first_rows_response": create_first_rows_response(hub_public_big, BIG_cols, BIG_rows),
            "parquet_response": None,
        },
    }
