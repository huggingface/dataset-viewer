# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from unittest.mock import patch

from huggingface_hub.hf_api import DatasetInfo
from libapi.exceptions import ResponseNotReadyError
from libapi.utils import get_cache_entry_from_steps
from libcommon.exceptions import (
    DatasetInBlockListError,
    NotSupportedDisabledRepositoryError,
    NotSupportedDisabledViewerError,
    NotSupportedPrivateRepositoryError,
)
from libcommon.operations import EntityInfo
from libcommon.processing_graph import processing_graph
from libcommon.queue import Queue
from libcommon.simple_cache import upsert_response
from pytest import raises

from api.config import AppConfig, EndpointConfig
from api.routes.endpoint import EndpointsDefinition


def test_endpoints_definition() -> None:
    endpoint_config = EndpointConfig.from_env()

    endpoints_definition = EndpointsDefinition(processing_graph, endpoint_config)
    assert endpoints_definition

    definition = endpoints_definition.steps_by_input_type_and_endpoint
    assert definition

    splits = definition["/splits"]
    assert splits is not None
    assert sorted(list(splits)) == ["config", "dataset"]
    assert splits["dataset"] is not None
    assert splits["config"] is not None
    assert len(splits["dataset"]) == 1  # Has one processing step
    assert len(splits["config"]) == 2  # Has two processing steps

    first_rows = definition["/first-rows"]
    assert first_rows is not None
    assert sorted(list(first_rows)) == ["split"]
    assert first_rows["split"] is not None
    assert len(first_rows["split"]) == 2  # Has two processing steps

    parquet = definition["/parquet"]
    assert parquet is not None
    assert sorted(list(parquet)) == ["config", "dataset"]
    assert parquet["dataset"] is not None
    assert parquet["config"] is not None
    assert len(parquet["dataset"]) == 1  # Only has one processing step
    assert len(parquet["config"]) == 1  # Only has one processing step

    dataset_info = definition["/info"]
    assert dataset_info is not None
    assert sorted(list(dataset_info)) == ["config", "dataset"]
    assert dataset_info["dataset"] is not None
    assert dataset_info["config"] is not None
    assert len(dataset_info["dataset"]) == 1  # Only has one processing step
    assert len(dataset_info["config"]) == 1  # Only has one processing step

    size = definition["/size"]
    assert size is not None
    assert sorted(list(size)) == ["config", "dataset"]
    assert size["dataset"] is not None
    assert size["config"] is not None
    assert len(size["dataset"]) == 1  # Only has one processing step
    assert len(size["config"]) == 1  # Only has one processing step

    opt_in_out_urls = definition["/opt-in-out-urls"]
    assert opt_in_out_urls is not None
    assert sorted(list(opt_in_out_urls)) == ["config", "dataset", "split"]
    assert opt_in_out_urls["split"] is not None
    assert opt_in_out_urls["config"] is not None
    assert opt_in_out_urls["dataset"] is not None
    assert len(opt_in_out_urls["split"]) == 1  # Only has one processing step
    assert len(opt_in_out_urls["config"]) == 1  # Only has one processing step
    assert len(opt_in_out_urls["dataset"]) == 1  # Only has one processing step

    is_valid = definition["/is-valid"]
    assert is_valid is not None
    assert sorted(list(is_valid)) == ["config", "dataset", "split"]
    assert is_valid["dataset"] is not None
    assert is_valid["config"] is not None
    assert is_valid["split"] is not None
    assert len(is_valid["dataset"]) == 1  # Only has one processing step
    assert len(is_valid["config"]) == 1  # Only has one processing step
    assert len(is_valid["split"]) == 1  # Only has one processing step

    # assert old deleted endpoints don't exist
    with raises(KeyError):
        _ = definition["/dataset-info"]
    with raises(KeyError):
        _ = definition["/parquet-and-dataset-info"]
    with raises(KeyError):
        _ = definition["/config-names"]


def test_get_cache_entry_from_steps() -> None:
    dataset = "dataset"
    revision = "revision"
    config = "config"

    app_config = AppConfig.from_env()

    cache_with_error = "config-split-names-from-streaming"
    cache_without_error = "config-split-names-from-info"

    upsert_response(
        kind=cache_without_error,
        dataset=dataset,
        dataset_git_revision=revision,
        config=config,
        content={},
        http_status=HTTPStatus.OK,
    )

    upsert_response(
        kind=cache_with_error,
        dataset=dataset,
        dataset_git_revision=revision,
        config=config,
        content={},
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    )

    # succeeded result is returned
    result = get_cache_entry_from_steps(
        processing_step_names=[cache_without_error, cache_with_error],
        dataset=dataset,
        config=config,
        split=None,
        hf_endpoint=app_config.common.hf_endpoint,
        blocked_datasets=[],
    )
    assert result
    assert result["http_status"] == HTTPStatus.OK

    # succeeded result is returned even if first step failed
    result = get_cache_entry_from_steps(
        processing_step_names=[cache_with_error, cache_without_error],
        dataset=dataset,
        config=config,
        split=None,
        hf_endpoint=app_config.common.hf_endpoint,
        blocked_datasets=[],
    )
    assert result
    assert result["http_status"] == HTTPStatus.OK

    # error result is returned if all steps failed
    result = get_cache_entry_from_steps(
        processing_step_names=[cache_with_error, cache_with_error],
        dataset=dataset,
        config=config,
        split=None,
        hf_endpoint=app_config.common.hf_endpoint,
        blocked_datasets=[],
    )
    assert result
    assert result["http_status"] == HTTPStatus.INTERNAL_SERVER_ERROR
    assert result["error_code"] is None

    # pending job throws exception
    queue = Queue()
    queue.add_job(job_type="dataset-split-names", dataset=dataset, revision=revision, config=config, difficulty=50)
    with patch("libcommon.operations.update_dataset", return_value=None):
        # ^ the dataset does not exist on the Hub, we don't want to raise an issue here
        with raises(ResponseNotReadyError):
            get_cache_entry_from_steps(
                processing_step_names=["dataset-split-names"],
                dataset=dataset,
                config=None,
                split=None,
                hf_endpoint=app_config.common.hf_endpoint,
                blocked_datasets=[],
            )


def test_get_cache_entry_from_steps_no_cache() -> None:
    dataset = "dataset"
    revision = "revision"
    config = "config"

    app_config = AppConfig.from_env()

    no_cache = "config-is-valid"

    with patch(
        "libcommon.operations.get_dataset_info",
        return_value=DatasetInfo(id=dataset, sha=revision, private=False, downloads=0, likes=0, tags=[]),
    ):
        # ^ the dataset does not exist on the Hub, we don't want to raise an issue here

        # if the dataset is not blocked and no cache exists, ResponseNotReadyError is raised
        with raises(ResponseNotReadyError):
            get_cache_entry_from_steps(
                processing_step_names=[no_cache],
                dataset=dataset,
                config=config,
                split=None,
                hf_endpoint=app_config.common.hf_endpoint,
                blocked_datasets=[],
            )


def test_get_cache_entry_from_steps_no_cache_disabled_viewer() -> None:
    dataset = "dataset"
    revision = "revision"
    config = "config"

    app_config = AppConfig.from_env()

    no_cache = "config-is-valid"

    with patch(
        "libcommon.operations.get_dataset_info",
        return_value=DatasetInfo(
            id=dataset, sha=revision, private=False, downloads=0, likes=0, tags=[], card_data={"viewer": False}
        ),
    ):
        # ^ the dataset does not exist on the Hub, we don't want to raise an issue here
        # we set the dataset as disabled

        with raises(NotSupportedDisabledViewerError):
            get_cache_entry_from_steps(
                processing_step_names=[no_cache],
                dataset=dataset,
                config=config,
                split=None,
                hf_endpoint=app_config.common.hf_endpoint,
                blocked_datasets=[],
            )


def test_get_cache_entry_from_steps_no_cache_disabled() -> None:
    dataset = "dataset"
    revision = "revision"
    config = "config"

    app_config = AppConfig.from_env()

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
            get_cache_entry_from_steps(
                processing_step_names=[no_cache],
                dataset=dataset,
                config=config,
                split=None,
                hf_endpoint=app_config.common.hf_endpoint,
                blocked_datasets=[],
            )


def test_get_cache_entry_from_steps_no_cache_private() -> None:
    dataset = "dataset"
    revision = "revision"
    config = "config"
    author = "author"

    app_config = AppConfig.from_env()

    no_cache = "config-is-valid"

    with patch(
        "libcommon.operations.get_dataset_info",
        return_value=DatasetInfo(id=dataset, sha=revision, private=True, downloads=0, likes=0, tags=[], author=author),
    ), patch("libcommon.operations.get_entity_info", return_value=EntityInfo(isPro=False, isEnterprise=False)):
        # ^ the dataset and the author do not exist on the Hub, we don't want to raise an issue here
        with raises(NotSupportedPrivateRepositoryError):
            get_cache_entry_from_steps(
                processing_step_names=[no_cache],
                dataset=dataset,
                config=config,
                split=None,
                hf_endpoint=app_config.common.hf_endpoint,
                blocked_datasets=[],
            )


def test_get_cache_entry_from_steps_no_cache_private_pro() -> None:
    dataset = "dataset"
    revision = "revision"
    config = "config"
    author = "author"

    app_config = AppConfig.from_env()

    no_cache = "config-is-valid"

    with patch(
        "libcommon.operations.get_dataset_info",
        return_value=DatasetInfo(id=dataset, sha=revision, private=True, downloads=0, likes=0, tags=[], author=author),
    ), patch("libcommon.operations.get_entity_info", return_value=EntityInfo(isPro=True, isEnterprise=False)):
        # ^ the dataset and the author do not exist on the Hub, we don't want to raise an issue here
        with raises(ResponseNotReadyError):
            get_cache_entry_from_steps(
                processing_step_names=[no_cache],
                dataset=dataset,
                config=config,
                split=None,
                hf_endpoint=app_config.common.hf_endpoint,
                blocked_datasets=[],
            )


def test_get_cache_entry_from_steps_no_cache_private_enterprise() -> None:
    dataset = "dataset"
    revision = "revision"
    config = "config"
    author = "author"

    app_config = AppConfig.from_env()

    no_cache = "config-is-valid"

    with patch(
        "libcommon.operations.get_dataset_info",
        return_value=DatasetInfo(id=dataset, sha=revision, private=True, downloads=0, likes=0, tags=[], author=author),
    ), patch("libcommon.operations.get_entity_info", return_value=EntityInfo(isPro=False, isEnterprise=True)):
        # ^ the dataset and the author do not exist on the Hub, we don't want to raise an issue here
        with raises(ResponseNotReadyError):
            get_cache_entry_from_steps(
                processing_step_names=[no_cache],
                dataset=dataset,
                config=config,
                split=None,
                hf_endpoint=app_config.common.hf_endpoint,
                blocked_datasets=[],
            )


def test_get_cache_entry_from_steps_no_cache_blocked() -> None:
    dataset = "dataset"
    revision = "revision"
    config = "config"

    app_config = AppConfig.from_env()

    no_cache = "config-is-valid"

    with patch(
        "libcommon.operations.get_dataset_info",
        return_value=DatasetInfo(id=dataset, sha=revision, private=False, downloads=0, likes=0, tags=[]),
    ):
        # ^ the dataset does not exist on the Hub, we don't want to raise an issue here

        # error result is returned if the dataset is blocked and no cache exists
        with raises(DatasetInBlockListError):
            get_cache_entry_from_steps(
                processing_step_names=[no_cache],
                dataset=dataset,
                config=config,
                split=None,
                hf_endpoint=app_config.common.hf_endpoint,
                blocked_datasets=[dataset],
            )
