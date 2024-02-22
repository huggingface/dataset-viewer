# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from unittest.mock import patch

from huggingface_hub.hf_api import DatasetInfo
from libcommon.exceptions import (
    DatasetInBlockListError,
    NotSupportedDisabledRepositoryError,
    NotSupportedDisabledViewerError,
    NotSupportedPrivateRepositoryError,
)
from libcommon.operations import EntityInfo
from libcommon.simple_cache import upsert_response
from pytest import raises

from libapi.exceptions import ResponseNotReadyError
from libapi.utils import get_cache_entry_from_step


def test_get_cache_entry_from_step(hf_endpoint: str) -> None:
    dataset = "dataset"
    revision = "revision"
    config = "config"

    kind = "config-split-names"

    upsert_response(
        kind=kind,
        dataset=dataset,
        dataset_git_revision=revision,
        config=config,
        content={},
        http_status=HTTPStatus.OK,
    )

    # success result is returned
    result = get_cache_entry_from_step(
        processing_step_name=kind,
        dataset=dataset,
        config=config,
        split=None,
        hf_endpoint=hf_endpoint,
        blocked_datasets=[],
    )
    assert result
    assert result["http_status"] == HTTPStatus.OK


def test_get_cache_entry_from_step_error(hf_endpoint: str) -> None:
    dataset = "dataset"
    revision = "revision"
    config = "config"

    kind = "config-split-names"

    upsert_response(
        kind=kind,
        dataset=dataset,
        dataset_git_revision=revision,
        config=config,
        content={},
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    )

    # error result
    result = get_cache_entry_from_step(
        processing_step_name=kind,
        dataset=dataset,
        config=config,
        split=None,
        hf_endpoint=hf_endpoint,
        blocked_datasets=[],
    )
    assert result
    assert result["http_status"] == HTTPStatus.INTERNAL_SERVER_ERROR
    assert result["error_code"] is None


def test_get_cache_entry_from_step_no_cache(hf_endpoint: str) -> None:
    dataset = "dataset"
    revision = "revision"
    config = "config"

    no_cache = "config-is-valid"

    with patch(
        "libcommon.operations.get_dataset_info",
        return_value=DatasetInfo(id=dataset, sha=revision, private=False, downloads=0, likes=0, tags=[]),
    ):
        # ^ the dataset does not exist on the Hub, we don't want to raise an issue here

        # if the dataset is not blocked and no cache exists, ResponseNotReadyError is raised
        with raises(ResponseNotReadyError):
            get_cache_entry_from_step(
                processing_step_name=no_cache,
                dataset=dataset,
                config=config,
                split=None,
                hf_endpoint=hf_endpoint,
                blocked_datasets=[],
            )


def test_get_cache_entry_from_step_no_cache_disabled_viewer(hf_endpoint: str) -> None:
    dataset = "dataset"
    revision = "revision"
    config = "config"

    no_cache = "config-is-valid"

    with patch(
        "libcommon.operations.get_dataset_info",
        return_value=DatasetInfo(
            id=dataset, sha=revision, private=False, downloads=0, likes=0, tags=[], card_data={"viewer": False}
        ),
    ):
        # ^ the dataset does not exist on the Hub, we don't want to raise an issue here

        with raises(NotSupportedDisabledViewerError):
            get_cache_entry_from_step(
                processing_step_name=no_cache,
                dataset=dataset,
                config=config,
                split=None,
                hf_endpoint=hf_endpoint,
                blocked_datasets=[],
            )


def test_get_cache_entry_from_step_no_cache_disabled(hf_endpoint: str) -> None:
    dataset = "dataset"
    revision = "revision"
    config = "config"

    no_cache = "config-is-valid"

    with patch(
        "libcommon.operations.get_dataset_info",
        return_value=DatasetInfo(
            id=dataset, sha=revision, private=False, downloads=0, likes=0, tags=[], disabled=True
        ),
    ):
        # ^ the dataset does not exist on the Hub, we don't want to raise an issue here
        # we set the dataset as disabled

        with raises(NotSupportedDisabledRepositoryError):
            get_cache_entry_from_step(
                processing_step_name=no_cache,
                dataset=dataset,
                config=config,
                split=None,
                hf_endpoint=hf_endpoint,
                blocked_datasets=[],
            )


def test_get_cache_entry_from_step_no_cache_private(hf_endpoint: str) -> None:
    dataset = "dataset"
    revision = "revision"
    config = "config"
    author = "author"

    no_cache = "config-is-valid"

    with patch(
        "libcommon.operations.get_dataset_info",
        return_value=DatasetInfo(id=dataset, sha=revision, private=True, downloads=0, likes=0, tags=[], author=author),
    ), patch("libcommon.operations.get_entity_info", return_value=EntityInfo(isPro=False, isEnterprise=False)):
        # ^ the dataset and the author do not exist on the Hub, we don't want to raise an issue here
        with raises(NotSupportedPrivateRepositoryError):
            get_cache_entry_from_step(
                processing_step_name=no_cache,
                dataset=dataset,
                config=config,
                split=None,
                hf_endpoint=hf_endpoint,
                blocked_datasets=[],
            )


def test_get_cache_entry_from_step_no_cache_private_pro(hf_endpoint: str) -> None:
    dataset = "dataset"
    revision = "revision"
    config = "config"
    author = "author"

    no_cache = "config-is-valid"

    with patch(
        "libcommon.operations.get_dataset_info",
        return_value=DatasetInfo(id=dataset, sha=revision, private=True, downloads=0, likes=0, tags=[], author=author),
    ), patch("libcommon.operations.get_entity_info", return_value=EntityInfo(isPro=True, isEnterprise=False)):
        # ^ the dataset and the author do not exist on the Hub, we don't want to raise an issue here
        with raises(ResponseNotReadyError):
            get_cache_entry_from_step(
                processing_step_name=no_cache,
                dataset=dataset,
                config=config,
                split=None,
                hf_endpoint=hf_endpoint,
                blocked_datasets=[],
            )


def test_get_cache_entry_from_step_no_cache_private_enterprise(hf_endpoint: str) -> None:
    dataset = "dataset"
    revision = "revision"
    config = "config"
    author = "author"

    no_cache = "config-is-valid"

    with patch(
        "libcommon.operations.get_dataset_info",
        return_value=DatasetInfo(id=dataset, sha=revision, private=True, downloads=0, likes=0, tags=[], author=author),
    ), patch("libcommon.operations.get_entity_info", return_value=EntityInfo(isPro=False, isEnterprise=True)):
        # ^ the dataset and the author do not exist on the Hub, we don't want to raise an issue here
        with raises(ResponseNotReadyError):
            get_cache_entry_from_step(
                processing_step_name=no_cache,
                dataset=dataset,
                config=config,
                split=None,
                hf_endpoint=hf_endpoint,
                blocked_datasets=[],
            )


def test_get_cache_entry_from_step_no_cache_blocked(hf_endpoint: str) -> None:
    dataset = "dataset"
    revision = "revision"
    config = "config"

    no_cache = "config-is-valid"

    with patch(
        "libcommon.operations.get_dataset_info",
        return_value=DatasetInfo(id=dataset, sha=revision, private=False, downloads=0, likes=0, tags=[]),
    ):
        # ^ the dataset does not exist on the Hub, we don't want to raise an issue here

        # error result is returned if the dataset is blocked and no cache exists
        with raises(DatasetInBlockListError):
            get_cache_entry_from_step(
                processing_step_name=no_cache,
                dataset=dataset,
                config=config,
                split=None,
                hf_endpoint=hf_endpoint,
                blocked_datasets=[dataset],
            )
