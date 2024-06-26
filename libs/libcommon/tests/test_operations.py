# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import time
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest
from huggingface_hub.hf_api import DatasetInfo, HfApi
from huggingface_hub.utils import HfHubHTTPError
from requests import Response

from libcommon.constants import CONFIG_SPLIT_NAMES_KIND, DATASET_CONFIG_NAMES_KIND, TAG_NFAA_SYNONYMS
from libcommon.exceptions import (
    DatasetInBlockListError,
    NotSupportedDisabledRepositoryError,
    NotSupportedPrivateRepositoryError,
    NotSupportedRepositoryNotFoundError,
    NotSupportedTagNFAAError,
)
from libcommon.operations import (
    CustomHfApi,
    backfill_dataset,
    delete_dataset,
    get_latest_dataset_revision_if_supported_or_raise,
    update_dataset,
)
from libcommon.orchestrator import finish_job
from libcommon.processing_graph import specification
from libcommon.queue.jobs import Queue
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import get_cache_entries_df, has_some_cache, upsert_response
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


@pytest.mark.parametrize(
    "name,expected_pro,expected_enterprise",
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


@contextmanager
def tmp_dataset(namespace: str, token: str, private: bool, tags: Optional[list[str]] = None) -> Iterator[str]:
    # create a test dataset in hub-ci, then delete it
    hf_api = HfApi(endpoint=CI_HUB_ENDPOINT, token=token)
    prefix = "private" if private else "public"
    dataset = f"{namespace}/{prefix}-dataset-{int(time.time() * 10e3)}"
    hf_api.create_repo(
        repo_id=dataset,
        private=private,
        repo_type="dataset",
    )
    if tags:
        README = "---\n" + "tags:\n" + "".join(f"- {tag}\n" for tag in tags) + "---"
        hf_api.upload_file(
            path_or_fileobj=README.encode(),
            path_in_repo="README.md",
            repo_id=dataset,
            token=token,
            repo_type="dataset",
        )
    try:
        yield dataset
    finally:
        try:
            hf_api.delete_repo(repo_id=dataset, repo_type="dataset")
        except Exception:
            # the dataset could already have been deleted
            pass


@pytest.mark.parametrize(
    "token,namespace",
    [
        (NORMAL_USER_TOKEN, NORMAL_USER),
        (NORMAL_USER_TOKEN, NORMAL_ORG),
        (ENTERPRISE_USER_TOKEN, ENTERPRISE_USER),
    ],
)
def test_get_revision_private_raises(token: str, namespace: str) -> None:
    with tmp_dataset(namespace=namespace, token=token, private=True) as dataset:
        with pytest.raises(NotSupportedPrivateRepositoryError):
            get_latest_dataset_revision_if_supported_or_raise(
                dataset=dataset, hf_endpoint=CI_HUB_ENDPOINT, hf_token=CI_APP_TOKEN
            )


@pytest.mark.parametrize(
    "token,namespace",
    [
        (PRO_USER_TOKEN, PRO_USER),
        (ENTERPRISE_USER_TOKEN, ENTERPRISE_ORG),
    ],
)
def test_get_revision_private(token: str, namespace: str) -> None:
    with tmp_dataset(namespace=namespace, token=token, private=True) as dataset:
        get_latest_dataset_revision_if_supported_or_raise(
            dataset=dataset, hf_endpoint=CI_HUB_ENDPOINT, hf_token=CI_APP_TOKEN
        )


@pytest.mark.parametrize(
    "token,namespace",
    [
        (NORMAL_USER_TOKEN, NORMAL_USER),
        (NORMAL_USER_TOKEN, NORMAL_ORG),
        (ENTERPRISE_USER_TOKEN, ENTERPRISE_USER),
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


@pytest.mark.parametrize(
    "tags,raises",
    [
        (None, False),
        ([], False),
        (["perfectly-fine-tag"], False),
        (["perfectly-fine-tag", "another-fine-tag"], False),
        ([TAG_NFAA_SYNONYMS[0]], True),
        ([TAG_NFAA_SYNONYMS[1]], True),
        (TAG_NFAA_SYNONYMS, True),
        (["perfectly-fine-tag", TAG_NFAA_SYNONYMS[0]], True),
    ],
)
def test_update_dataset_with_nfaa_tag_raises(
    queue_mongo_resource: QueueMongoResource,
    cache_mongo_resource: CacheMongoResource,
    tags: Optional[list[str]],
    raises: bool,
) -> None:
    with tmp_dataset(namespace=NORMAL_USER, token=NORMAL_USER_TOKEN, private=False, tags=tags) as dataset:
        if raises:
            with pytest.raises(NotSupportedTagNFAAError):
                update_dataset(dataset=dataset, hf_endpoint=CI_HUB_ENDPOINT, hf_token=CI_APP_TOKEN)
        else:
            update_dataset(dataset=dataset, hf_endpoint=CI_HUB_ENDPOINT, hf_token=CI_APP_TOKEN)


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
    "tags",
    [
        None,
        [],
        ["perfectly-fine-tag"],
        TAG_NFAA_SYNONYMS,
    ],
    # ^ for Pro and Enterprise Hub, *private* NFAA datasets are allowed
)
@pytest.mark.parametrize(
    "token,namespace",
    [
        (PRO_USER_TOKEN, PRO_USER),
        (ENTERPRISE_USER_TOKEN, ENTERPRISE_ORG),
    ],
)
def test_update_private(
    queue_mongo_resource: QueueMongoResource,
    cache_mongo_resource: CacheMongoResource,
    tags: Optional[list[str]],
    token: str,
    namespace: str,
) -> None:
    with tmp_dataset(namespace=namespace, token=token, private=True, tags=tags) as dataset:
        update_dataset(dataset=dataset, hf_endpoint=CI_HUB_ENDPOINT, hf_token=CI_APP_TOKEN)


@pytest.mark.parametrize(
    "tags,raises",
    [
        (None, False),
        ([], False),
        (["perfectly-fine-tag"], False),
        (TAG_NFAA_SYNONYMS, True),
    ],
    # ^ for Pro and Enterprise Hub, *public* NFAA datasets are disallowed
)
@pytest.mark.parametrize(
    "token,namespace",
    [
        (PRO_USER_TOKEN, PRO_USER),
        (ENTERPRISE_USER_TOKEN, ENTERPRISE_ORG),
    ],
)
def test_update_public_nfaa(
    queue_mongo_resource: QueueMongoResource,
    cache_mongo_resource: CacheMongoResource,
    tags: Optional[list[str]],
    raises: bool,
    token: str,
    namespace: str,
) -> None:
    with tmp_dataset(namespace=namespace, token=token, private=False, tags=tags) as dataset:
        if raises:
            with pytest.raises(NotSupportedTagNFAAError):
                update_dataset(dataset=dataset, hf_endpoint=CI_HUB_ENDPOINT, hf_token=CI_APP_TOKEN)
        else:
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
        storage_root=str(tmp_path / "assets"),
        base_url="https://example.com/assets",
    )
    cached_assets_storage_client = StorageClient(
        protocol="file",
        storage_root=str(tmp_path / "cached-assets"),
        base_url="https://example.com/cached-assets",
    )

    if create_assets:
        assets_storage_client._fs.mkdirs(dataset, exist_ok=True)
        assets_storage_client._fs.touch(assets_storage_client.get_full_path(image_key))
        assert assets_storage_client.exists(image_key)

    if create_cached_assets:
        cached_assets_storage_client._fs.mkdirs(dataset, exist_ok=True)
        cached_assets_storage_client._fs.touch(cached_assets_storage_client.get_full_path(image_key))
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


def test_2274_only_first_steps(
    queue_mongo_resource: QueueMongoResource,
    cache_mongo_resource: CacheMongoResource,
) -> None:
    JOB_RUNNER_VERSION = 1
    FIRST_CACHE_KINDS = ["dataset-config-names", "dataset-filetypes"]

    # see https://github.com/huggingface/dataset-viewer/issues/2274
    with tmp_dataset(namespace=NORMAL_USER, token=NORMAL_USER_TOKEN, private=False) as dataset:
        queue = Queue()

        # first update: two jobs are created
        update_dataset(dataset=dataset, hf_endpoint=CI_HUB_ENDPOINT, hf_token=CI_APP_TOKEN)

        pending_jobs_df = queue.get_pending_jobs_df(dataset=dataset)
        cache_entries_df = get_cache_entries_df(dataset=dataset)

        assert len(pending_jobs_df) == 2
        assert pending_jobs_df.iloc[0]["type"] in FIRST_CACHE_KINDS
        assert cache_entries_df.empty

        # process the first two jobs
        for _ in range(2):
            finish_job(
                job_result={
                    "job_info": queue.start_job(),
                    "job_runner_version": JOB_RUNNER_VERSION,
                    "is_success": True,
                    "output": {
                        "content": {},
                        "http_status": HTTPStatus.OK,
                        "error_code": None,
                        "details": None,
                        "progress": 1.0,
                    },
                }
            )

        assert len(queue.get_pending_jobs_df(dataset=dataset)) == 8
        assert len(get_cache_entries_df(dataset=dataset)) == 2

        # let's delete all the jobs, to get in the same state as the bug
        queue.delete_dataset_waiting_jobs(dataset=dataset)

        assert queue.get_pending_jobs_df(dataset=dataset).empty
        assert len(get_cache_entries_df(dataset=dataset)) == 2

        # try to reproduce the bug in #2375 by update at this point: no
        update_dataset(dataset=dataset, hf_endpoint=CI_HUB_ENDPOINT, hf_token=CI_APP_TOKEN)

        assert queue.get_pending_jobs_df(dataset=dataset).empty
        assert len(get_cache_entries_df(dataset=dataset)) == 2

        # THE BUG IS REPRODUCED: the jobs should have been recreated at that point.

        backfill_dataset(dataset=dataset, hf_endpoint=CI_HUB_ENDPOINT, hf_token=CI_APP_TOKEN)

        assert not queue.get_pending_jobs_df(dataset=dataset).empty
        assert len(get_cache_entries_df(dataset=dataset)) == 2


@pytest.mark.parametrize(
    "config_names,should_be_recreated",
    [(["config_ok"], False), (["config_nok"], True), (["config_ok", "config_nok"], True)],
)
def test_2767_delete_inexistent_configs(
    queue_mongo_resource: QueueMongoResource,
    cache_mongo_resource: CacheMongoResource,
    config_names: list[str],
    should_be_recreated: bool,
) -> None:
    # see https://github.com/huggingface/dataset-viewer/issues/2767
    with tmp_dataset(namespace=NORMAL_USER, token=NORMAL_USER_TOKEN, private=False) as dataset:
        revision = get_latest_dataset_revision_if_supported_or_raise(
            dataset=dataset,
            hf_endpoint=CI_HUB_ENDPOINT,
            hf_token=NORMAL_USER_TOKEN,
        )
        upsert_response(
            kind=DATASET_CONFIG_NAMES_KIND,
            dataset=dataset,
            content={"config_names": [{"dataset": dataset, "config": "config_ok"}]},
            http_status=HTTPStatus.OK,
            job_runner_version=specification[DATASET_CONFIG_NAMES_KIND]["job_runner_version"],
            dataset_git_revision=revision,
        )
        for config_name in config_names:
            upsert_response(
                kind=CONFIG_SPLIT_NAMES_KIND,
                dataset=dataset,
                config=config_name,
                content={"not": "important"},
                http_status=HTTPStatus.OK,
                job_runner_version=specification[CONFIG_SPLIT_NAMES_KIND]["job_runner_version"],
                dataset_git_revision=revision,
            )
        operation_statistics = backfill_dataset(dataset=dataset, hf_endpoint=CI_HUB_ENDPOINT, hf_token=CI_APP_TOKEN)
        assert operation_statistics.num_backfilled_datasets == 1
        # ^ when the config names contain something else than "config_ok", the dataset should have been deleted in the cache,
        # and the first job should have been created again
        if should_be_recreated:
            assert Queue().has_pending_jobs(dataset=dataset, job_types=[DATASET_CONFIG_NAMES_KIND])
            assert get_cache_entries_df(dataset=dataset).empty
        else:
            assert not Queue().has_pending_jobs(dataset=dataset, job_types=[DATASET_CONFIG_NAMES_KIND])
            assert len(get_cache_entries_df(dataset=dataset)) == len(config_names) + 1


@pytest.mark.parametrize(
    "split_names,should_be_recreated",
    [(["split_ok"], False), (["split_nok"], True), (["split_ok", "split_nok"], True)],
)
def test_2767_delete_inexistent_splits(
    queue_mongo_resource: QueueMongoResource,
    cache_mongo_resource: CacheMongoResource,
    split_names: list[str],
    should_be_recreated: bool,
) -> None:
    # see https://github.com/huggingface/dataset-viewer/issues/2767
    with tmp_dataset(namespace=NORMAL_USER, token=NORMAL_USER_TOKEN, private=False) as dataset:
        config = "config"
        revision = get_latest_dataset_revision_if_supported_or_raise(
            dataset=dataset,
            hf_endpoint=CI_HUB_ENDPOINT,
            hf_token=NORMAL_USER_TOKEN,
        )
        upsert_response(
            kind=DATASET_CONFIG_NAMES_KIND,
            dataset=dataset,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
            job_runner_version=specification[DATASET_CONFIG_NAMES_KIND]["job_runner_version"],
            dataset_git_revision=revision,
        )
        upsert_response(
            kind=CONFIG_SPLIT_NAMES_KIND,
            dataset=dataset,
            config=config,
            content={"splits": [{"dataset": dataset, "config": config, "split": "split_ok"}]},
            http_status=HTTPStatus.OK,
            job_runner_version=specification[CONFIG_SPLIT_NAMES_KIND]["job_runner_version"],
            dataset_git_revision=revision,
        )
        for split_name in split_names:
            upsert_response(
                kind="split-first-rows",
                dataset=dataset,
                config=config,
                split=split_name,
                content={"not": "important"},
                http_status=HTTPStatus.OK,
                job_runner_version=specification["split-first-rows"]["job_runner_version"],
                dataset_git_revision=revision,
            )
        operation_statistics = backfill_dataset(dataset=dataset, hf_endpoint=CI_HUB_ENDPOINT, hf_token=CI_APP_TOKEN)
        assert operation_statistics.num_backfilled_datasets == 1
        # ^ when the split names contain something else than "split_ok", the dataset should have been deleted in the cache,
        # and the first job should have been created again
        if should_be_recreated:
            assert Queue().has_pending_jobs(dataset=dataset, job_types=[DATASET_CONFIG_NAMES_KIND])
            assert get_cache_entries_df(dataset=dataset).empty
        else:
            assert not Queue().has_pending_jobs(dataset=dataset, job_types=[DATASET_CONFIG_NAMES_KIND])
            assert len(get_cache_entries_df(dataset=dataset)) == len(split_names) + 2
