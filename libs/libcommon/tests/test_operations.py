# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import time
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Optional

import pytest
from huggingface_hub import HfApi

from libcommon.operations import CustomHfApi, delete_dataset, get_dataset_status, update_dataset
from libcommon.queue import Queue
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import has_some_cache, upsert_response
from libcommon.storage_client import StorageClient
from libcommon.utils import SupportStatus

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
def test_get_dataset_status() -> None:
    get_dataset_status(dataset=PROD_PUBLIC_DATASET, hf_endpoint=PROD_HUB_ENDPOINT)


@pytest.mark.real_dataset
def test_get_dataset_status_timeout() -> None:
    with pytest.raises(Exception):
        get_dataset_status(dataset=PROD_PUBLIC_DATASET, hf_endpoint=PROD_HUB_ENDPOINT, hf_timeout_seconds=0.01)


@pytest.mark.parametrize(
    "name, expected_pro, expected_enterprise",
    [
        (NORMAL_USER, False, None),
        (PRO_USER, True, None),
        (NORMAL_ORG, None, False),
        (ENTERPRISE_ORG, None, True),
    ],
)
def test_whoisthis(name: str, expected_pro: Optional[bool], expected_enterprise: Optional[bool]) -> None:
    entity_info = CustomHfApi(endpoint=CI_HUB_ENDPOINT).whoisthis(name=name, token=CI_APP_TOKEN)
    assert entity_info.is_pro == expected_pro
    assert entity_info.is_enterprise == expected_enterprise


# TODO: use a CI dataset instead of a prod dataset
@pytest.mark.real_dataset
@pytest.mark.parametrize(
    "blocked_datasets,expected_status",
    [
        ([], SupportStatus.PUBLIC),
        (["dummy"], SupportStatus.PUBLIC),
        ([f"{PROD_PUBLIC_DATASET}-other*"], SupportStatus.PUBLIC),
        ([PROD_PUBLIC_DATASET], SupportStatus.UNSUPPORTED),
        ([f"{PROD_PUBLIC_USER}/*"], SupportStatus.UNSUPPORTED),
        ([f"{PROD_PUBLIC_USER}/{PROD_PUBLIC_REPO[:2]}*"], SupportStatus.UNSUPPORTED),
        ([PROD_PUBLIC_DATASET, "dummy"], SupportStatus.UNSUPPORTED),
        (["dummy", PROD_PUBLIC_DATASET], SupportStatus.UNSUPPORTED),
    ],
)
def test_get_dataset_status_blocked_datasets(blocked_datasets: list[str], expected_status: SupportStatus) -> None:
    assert (
        get_dataset_status(
            dataset=PROD_PUBLIC_DATASET, hf_endpoint=PROD_HUB_ENDPOINT, blocked_datasets=blocked_datasets
        ).support_status
        == expected_status
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
    "token,namespace,expected_status",
    [
        (NORMAL_USER_TOKEN, NORMAL_USER, SupportStatus.UNSUPPORTED),
        (NORMAL_USER_TOKEN, NORMAL_ORG, SupportStatus.UNSUPPORTED),
        (PRO_USER_TOKEN, PRO_USER, SupportStatus.PRO_USER),
        (ENTERPRISE_USER_TOKEN, ENTERPRISE_USER, SupportStatus.UNSUPPORTED),
        (ENTERPRISE_USER_TOKEN, ENTERPRISE_ORG, SupportStatus.ENTERPRISE_ORG),
    ],
)
def test_get_dataset_status_private(token: str, namespace: str, expected_status: SupportStatus) -> None:
    with tmp_dataset(namespace=namespace, token=token, private=True) as dataset:
        assert (
            get_dataset_status(dataset=dataset, hf_endpoint=CI_HUB_ENDPOINT, hf_token=CI_APP_TOKEN).support_status
            == expected_status
        )


@pytest.mark.parametrize(
    "token,namespace,private,expected_creation",
    [
        (NORMAL_USER_TOKEN, NORMAL_USER, False, True),
        (NORMAL_USER_TOKEN, NORMAL_USER, True, False),
        (NORMAL_USER_TOKEN, NORMAL_ORG, True, False),
        (PRO_USER_TOKEN, PRO_USER, True, True),
        (ENTERPRISE_USER_TOKEN, ENTERPRISE_USER, True, False),
        (ENTERPRISE_USER_TOKEN, ENTERPRISE_ORG, True, True),
    ],
)
def test_update(
    queue_mongo_resource: QueueMongoResource,
    cache_mongo_resource: CacheMongoResource,
    token: str,
    namespace: str,
    private: bool,
    expected_creation: bool,
) -> None:
    with tmp_dataset(namespace=namespace, token=token, private=private) as dataset:
        assert (
            update_dataset(dataset=dataset, cache_max_days=1, hf_endpoint=CI_HUB_ENDPOINT, hf_token=CI_APP_TOKEN)
            == expected_creation
        )
        if expected_creation:
            assert Queue().has_pending_jobs(dataset=dataset)
            # delete the dataset by adding it to the blocked list
            assert not update_dataset(
                dataset=dataset,
                blocked_datasets=[dataset],
                cache_max_days=1,
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
