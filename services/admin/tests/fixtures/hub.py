# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

# Adapted from https://github.com/huggingface/datasets/blob/main/tests/fixtures/hub.py

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from typing import Literal, Optional, TypedDict, cast

import pytest
import requests
from huggingface_hub import HfApi

# see https://github.com/huggingface/moon-landing/blob/main/server/scripts/staging-seed-db.ts
CI_HUB_USER = "DVUser"
CI_HUB_USER_API_TOKEN = "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"

CI_HUB_ENDPOINT = "https://hub-ci.huggingface.co"


@pytest.fixture(scope="session")
def hf_api() -> HfApi:
    return HfApi(endpoint=CI_HUB_ENDPOINT)


@pytest.fixture(scope="session")
def hf_endpoint() -> str:
    return CI_HUB_ENDPOINT


@pytest.fixture(scope="session")
def hf_token() -> str:
    return CI_HUB_USER_API_TOKEN


@pytest.fixture
def cleanup_repo(hf_api: HfApi) -> Callable[[str], None]:
    def _cleanup_repo(repo_id: str) -> None:
        hf_api.delete_repo(repo_id=repo_id, token=CI_HUB_USER_API_TOKEN, repo_type="dataset")

    return _cleanup_repo


@pytest.fixture
def temporary_repo(cleanup_repo: Callable[[str], None]) -> Callable[[str], Iterator[str]]:
    @contextmanager
    def _temporary_repo(repo_id: str) -> Iterator[str]:
        try:
            yield repo_id
        finally:
            cleanup_repo(repo_id)

    return _temporary_repo  # type: ignore


def create_unique_repo_name(prefix: str, user: str) -> str:
    repo_name = f"{prefix}-{int(time.time() * 10e3)}"
    return f"{user}/{repo_name}"


def create_hf_dataset_repo(
    hf_api: HfApi,
    hf_token: str,
    prefix: str,
    *,
    private: bool = False,
    gated: Optional[str] = None,
    user: str = CI_HUB_USER,
) -> str:
    repo_id = create_unique_repo_name(prefix, user)
    hf_api.create_repo(repo_id=repo_id, token=hf_token, repo_type="dataset", private=private)
    if gated:
        hf_api.update_repo_settings(
            repo_id=repo_id,
            token=hf_token,
            gated=cast(Literal["auto", "manual", False], gated),
            repo_type="dataset",
        )
    return repo_id


# https://docs.pytest.org/en/6.2.x/fixture.html#yield-fixtures-recommended
@pytest.fixture(scope="session", autouse=True)
def hf_public_dataset_repo_empty(hf_api: HfApi, hf_token: str) -> Iterator[str]:
    repo_id = create_hf_dataset_repo(hf_api=hf_api, hf_token=hf_token, prefix="repo_empty")
    yield repo_id
    with suppress(requests.exceptions.HTTPError, ValueError):
        hf_api.delete_repo(repo_id=repo_id, token=hf_token, repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def hf_gated_dataset_repo_empty(hf_api: HfApi, hf_token: str) -> Iterator[str]:
    repo_id = create_hf_dataset_repo(hf_api=hf_api, hf_token=hf_token, prefix="repo_empty", gated="auto")
    yield repo_id
    with suppress(requests.exceptions.HTTPError, ValueError):
        hf_api.delete_repo(repo_id=repo_id, token=hf_token, repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def hf_private_dataset_repo_empty(hf_api: HfApi, hf_token: str) -> Iterator[str]:
    repo_id = create_hf_dataset_repo(hf_api=hf_api, hf_token=hf_token, prefix="repo_empty", private=True)
    yield repo_id
    with suppress(requests.exceptions.HTTPError, ValueError):
        hf_api.delete_repo(repo_id=repo_id, token=hf_token, repo_type="dataset")


class DatasetRepos(TypedDict):
    public: str
    private: str
    gated: str


DatasetReposType = Literal["public", "private", "gated"]


@pytest.fixture(scope="session", autouse=True)
def hf_dataset_repos_csv_data(
    hf_public_dataset_repo_empty: str,
    hf_gated_dataset_repo_empty: str,
    hf_private_dataset_repo_empty: str,
) -> DatasetRepos:
    return {
        "public": hf_public_dataset_repo_empty,
        "private": hf_private_dataset_repo_empty,
        "gated": hf_gated_dataset_repo_empty,
    }
