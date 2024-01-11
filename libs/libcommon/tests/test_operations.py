# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import time
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

import pytest
from huggingface_hub.hf_api import DatasetInfo, HfApi
from huggingface_hub.utils import HfHubHTTPError
from requests import Response  # type: ignore

from libcommon.exceptions import (
    DatasetInBlockListError,
    NotSupportedDisabledRepositoryError,
    NotSupportedPrivateRepositoryError,
    NotSupportedRepositoryNotFoundError,
)
from libcommon.operations import delete_dataset, get_latest_dataset_revision_if_supported_or_raise, update_dataset
from libcommon.queue import Queue
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import has_some_cache, upsert_response
from libcommon.storage_client import StorageClient

from .constants import (
    CI_APP_TOKEN,
    CI_HUB_ENDPOINT,
    ENTERPRISE_ORG,
    ENTERPRISE_USER,
    ENTERPRISE_USER_TOKEN,
    NORMAL_ORG,
    NORMAL_USER,
    NORMAL_USER_TOKEN,
    PRO_USER,
    PRO_USER_TOKEN,
    PROD_HUB_ENDPOINT,
)
from .utils import REVISION_NAME

PROD_PUBLIC_USER = "severo"
PROD_PUBLIC_REPO = "glue"
PROD_PUBLIC_DATASET = f"{PROD_PUBLIC_USER}/{PROD_PUBLIC_REPO}"


@pytest.mark.real_dataset
def test_get_revision() -> None:
    get_latest_dataset_revision_if_supported_or_raise(dataset=PROD_PUBLIC_DATASET, hf_endpoint=PROD_HUB_ENDPOINT)


@pytest.mark.real_dataset
def test_get_revision_timeout() -> None:
    with pytest.raises(Exception):
        get_latest_dataset_revision_if_supported_or_raise(
            dataset=PROD_PUBLIC_DATASET, hf_endpoint=PROD_HUB_ENDPOINT, hf_timeout_seconds=0.01
        )


@contextmanager
def tmp_dataset(namespace: str, token: str, private: bool) -> Iterator[str]:
    # create a test dataset in hub-ci, then delete it
    hf_api = HfApi(endpoint=CI_HUB_ENDPOINT, token=token)
    dataset = f"{namespace}/private-dataset-{int(time.time() * 10e3)}"
    hf_api.create_repo(
        repo_id=dataset,
        private=private,
        repo_type="dataset",
    )
    try:
        yield dataset
    finally:
        hf_api.delete_repo(repo_id=dataset, repo_type="dataset")


@pytest.mark.parametrize(
    "token,namespace",
    [
        (NORMAL_USER_TOKEN, NORMAL_USER),
        (NORMAL_USER_TOKEN, NORMAL_ORG),
        (PRO_USER_TOKEN, PRO_USER),
        (ENTERPRISE_USER_TOKEN, ENTERPRISE_USER),
        (ENTERPRISE_USER_TOKEN, ENTERPRISE_ORG),
    ],
)
def test_get_revision_private(token: str, namespace: str) -> None:
    with tmp_dataset(namespace=namespace, token=token, private=True) as dataset:
        with pytest.raises(NotSupportedPrivateRepositoryError):
            get_latest_dataset_revision_if_supported_or_raise(
                dataset=dataset, hf_endpoint=CI_HUB_ENDPOINT, hf_token=CI_APP_TOKEN
            )


@pytest.mark.parametrize(
    "token,namespace",
    [
        (NORMAL_USER_TOKEN, NORMAL_USER),
        (NORMAL_USER_TOKEN, NORMAL_ORG),
        (PRO_USER_TOKEN, PRO_USER),
        (ENTERPRISE_USER_TOKEN, ENTERPRISE_USER),
        (ENTERPRISE_USER_TOKEN, ENTERPRISE_ORG),
    ],
)
def test_update_private_raises(
    queue_mongo_resource: QueueMongoResource,
    cache_mongo_resource: CacheMongoResource,
    token: str,
    namespace: str,
) -> None:
    with tmp_dataset(namespace=namespace, token=token, private=True) as dataset:
        with pytest.raises(NotSupportedPrivateRepositoryError):
            update_dataset(dataset=dataset, hf_endpoint=CI_HUB_ENDPOINT, hf_token=CI_APP_TOKEN)


def test_update_non_existent_raises(
    queue_mongo_resource: QueueMongoResource,
    cache_mongo_resource: CacheMongoResource,
) -> None:
    with pytest.raises(NotSupportedRepositoryNotFoundError):
        update_dataset(dataset="this-dataset-does-not-exists", hf_endpoint=CI_HUB_ENDPOINT, hf_token=CI_APP_TOKEN)


def test_update_disabled_dataset_raises_way_1(
    queue_mongo_resource: QueueMongoResource,
    cache_mongo_resource: CacheMongoResource,
) -> None:
    dataset = "this-dataset-is-disabled"
    with patch(
        "libcommon.operations.get_dataset_info",
        return_value=DatasetInfo(
            id=dataset, sha="not-important", private=False, downloads=0, likes=0, tags=[], disabled=True
        ),
    ):
        # ^ we have no programmatical way to disable a dataset, so we mock the response of the API
        # possibly, the API can raise a 403 error code instead of returning a disabled field
        # see following test
        with pytest.raises(NotSupportedDisabledRepositoryError):
            update_dataset(dataset=dataset, hf_endpoint=CI_HUB_ENDPOINT, hf_token=CI_APP_TOKEN)


def test_update_disabled_dataset_raises_way_2(
    queue_mongo_resource: QueueMongoResource,
    cache_mongo_resource: CacheMongoResource,
) -> None:
    dataset = "this-dataset-is-disabled"

    response = Response()
    response.status_code = 403
    response.headers["X-Error-Message"] = "Access to this resource is disabled."
    with patch(
        "libcommon.operations.get_dataset_info",
        raise_exception=HfHubHTTPError("some message", response=response),
    ):
        # ^ we have no programmatical way to disable a dataset, so we mock the response of the API
        with pytest.raises(NotSupportedDisabledRepositoryError):
            update_dataset(dataset=dataset, hf_endpoint=CI_HUB_ENDPOINT, hf_token=CI_APP_TOKEN)


@pytest.mark.parametrize(
    "token,namespace",
    [
        (NORMAL_USER_TOKEN, NORMAL_USER),
    ],
)
def test_update_public_does_not_raise(
    queue_mongo_resource: QueueMongoResource,
    cache_mongo_resource: CacheMongoResource,
    token: str,
    namespace: str,
) -> None:
    with tmp_dataset(namespace=namespace, token=token, private=False) as dataset:
        update_dataset(dataset=dataset, hf_endpoint=CI_HUB_ENDPOINT, hf_token=CI_APP_TOKEN)
        assert Queue().has_pending_jobs(dataset=dataset)
        # delete the dataset by adding it to the blocked list
        with pytest.raises(DatasetInBlockListError):
            update_dataset(
                dataset=dataset,
                blocked_datasets=[dataset],
                hf_endpoint=CI_HUB_ENDPOINT,
                hf_token=CI_APP_TOKEN,
            )
        assert not Queue().has_pending_jobs(dataset=dataset)


@pytest.mark.parametrize(
    "create_assets,create_cached_assets",
    [
        (True, True),  # delete dataset with assets and cached assets
        (False, True),  # delete dataset with assets
        (True, False),  # delete dataset with cached assets
        (False, False),  # delete dataset without assets or cached assets
    ],
)
def test_delete_obsolete_cache(
    queue_mongo_resource: QueueMongoResource,
    cache_mongo_resource: CacheMongoResource,
    create_assets: bool,
    create_cached_assets: bool,
    tmp_path: Path,
) -> None:
    dataset = "dataset"
    image_key = f"{dataset}/image.jpg"

    assets_storage_client = StorageClient(
        protocol="file",
        root=str(tmp_path),
        folder="assets",
    )
    cached_assets_storage_client = StorageClient(
        protocol="file",
        root=str(tmp_path),
        folder="cached-assets",
    )

    if create_assets:
        assets_storage_client._fs.mkdirs(dataset, exist_ok=True)
        assets_storage_client._fs.touch(f"{assets_storage_client.get_base_directory()}/{image_key}")
        assert assets_storage_client.exists(image_key)

    if create_cached_assets:
        cached_assets_storage_client._fs.mkdirs(dataset, exist_ok=True)
        cached_assets_storage_client._fs.touch(f"{cached_assets_storage_client.get_base_directory()}/{image_key}")
        assert cached_assets_storage_client.exists(image_key)

    upsert_response(
        kind="kind_1",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        content={"config_names": [{"dataset": dataset, "config": "config"}]},
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind="kind_2",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config="config",
        content={"splits": [{"dataset": dataset, "config": "config", "split": "split"}]},
        http_status=HTTPStatus.OK,
    )
    Queue().add_job(job_type="type_3", dataset=dataset, revision=REVISION_NAME, difficulty=50)

    assert has_some_cache(dataset=dataset)
    assert Queue().has_pending_jobs(dataset=dataset)

    delete_dataset(dataset=dataset, storage_clients=[cached_assets_storage_client, assets_storage_client])

    assert not assets_storage_client.exists(image_key)
    assert not cached_assets_storage_client.exists(image_key)
    assert not has_some_cache(dataset=dataset)
    assert not Queue().has_pending_jobs(dataset=dataset)
