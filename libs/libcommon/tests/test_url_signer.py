# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from collections.abc import Mapping
from copy import deepcopy
from typing import Literal

import pytest

from libcommon.dtos import RowsContent
from libcommon.storage_client import StorageClient
from libcommon.url_signer import URLSigner, get_asset_url_paths, to_features_dict
from libcommon.viewer_utils.features import to_features_list
from libcommon.viewer_utils.rows import create_first_rows_response

from .constants import (
    DATASETS_NAMES,
    DEFAULT_COLUMNS_MAX_NUMBER,
    DEFAULT_CONFIG,
    DEFAULT_MIN_CELL_BYTES,
    DEFAULT_REVISION,
    DEFAULT_ROWS_MIN_NUMBER,
    DEFAULT_SPLIT,
    SOME_BYTES,
)
from .types import DatasetFixture
from .utils import get_dataset_rows_content


@pytest.mark.parametrize("dataset_name", DATASETS_NAMES)
def test_to_features_dict(datasets_fixtures: Mapping[str, DatasetFixture], dataset_name: str) -> None:
    datasets_fixture = datasets_fixtures[dataset_name]
    features = datasets_fixture.dataset.features
    features_list = to_features_list(features)
    features_dict = to_features_dict(features_list)
    assert isinstance(features_dict, dict)
    assert len(features_dict) > 0
    assert features.to_dict() == features_dict


@pytest.mark.parametrize("dataset_name", DATASETS_NAMES)
def test_get_asset_url_paths(datasets_fixtures: Mapping[str, DatasetFixture], dataset_name: str) -> None:
    dataset_fixture = datasets_fixtures[dataset_name]
    asset_url_paths = get_asset_url_paths(dataset_fixture.dataset.features)
    assert asset_url_paths == dataset_fixture.expected_asset_url_paths


FAKE_SIGNING_PREFIX = "?signed"


class FakeUrlSigner(URLSigner):
    def __init__(self) -> None:
        self.counter = 0

    def sign_url(self, url: str) -> str:
        self.counter += 1
        return url + FAKE_SIGNING_PREFIX


@pytest.mark.parametrize("dataset_name", DATASETS_NAMES)
def test__sign_asset_url_path_in_place(datasets_fixtures: Mapping[str, DatasetFixture], dataset_name: str) -> None:
    dataset_fixture = datasets_fixtures[dataset_name]
    url_signer = FakeUrlSigner()
    for asset_url_path in dataset_fixture.expected_asset_url_paths:
        cell_asset_url_path = asset_url_path.enter()
        # ^ remove the column name, as we will sign the cell, not the row
        url_signer._sign_asset_url_path_in_place(
            cell=deepcopy(dataset_fixture.expected_cell), asset_url_path=cell_asset_url_path
        )

    assert url_signer.counter == dataset_fixture.expected_num_asset_urls


@pytest.mark.parametrize("dataset_name", DATASETS_NAMES)
def test__get_asset_url_paths_from_first_rows(
    storage_client: StorageClient, datasets_fixtures: Mapping[str, DatasetFixture], dataset_name: str
) -> None:
    dataset_fixture = datasets_fixtures[dataset_name]
    dataset = dataset_fixture.dataset

    # no need for the rows in this test
    def get_fake_rows_content(rows_max_number: int) -> RowsContent:  # noqa: ARG001
        return RowsContent(rows=[], all_fetched=False)

    first_rows = create_first_rows_response(
        dataset=dataset_name,
        revision=DEFAULT_REVISION,
        config=DEFAULT_CONFIG,
        split=DEFAULT_SPLIT,
        storage_client=storage_client,
        features=dataset.features,
        get_rows_content=get_fake_rows_content,
        min_cell_bytes=0,
        rows_max_bytes=1000000,
        rows_max_number=1000000,
        rows_min_number=0,
        columns_max_number=100000,
    )

    url_signer = FakeUrlSigner()
    asset_url_paths = url_signer._get_asset_url_paths_from_first_rows(first_rows=first_rows)

    assert asset_url_paths == dataset_fixture.expected_asset_url_paths


@pytest.mark.parametrize("dataset_name", DATASETS_NAMES)
def test_sign_urls_in_first_rows_in_place(
    storage_client: StorageClient, datasets_fixtures: Mapping[str, DatasetFixture], dataset_name: str
) -> None:
    dataset_fixture = datasets_fixtures[dataset_name]
    dataset = dataset_fixture.dataset

    first_rows = create_first_rows_response(
        dataset=dataset_name,
        revision=DEFAULT_REVISION,
        config=DEFAULT_CONFIG,
        split=DEFAULT_SPLIT,
        storage_client=storage_client,
        features=dataset.features,
        get_rows_content=get_dataset_rows_content(dataset=dataset),
        min_cell_bytes=0,
        rows_max_bytes=1000000,
        rows_max_number=1000000,
        rows_min_number=0,
        columns_max_number=100000,
    )

    url_signer = FakeUrlSigner()
    url_signer.sign_urls_in_first_rows_in_place(first_rows=first_rows)

    assert url_signer.counter == dataset_fixture.expected_num_asset_urls


@pytest.mark.parametrize(
    "dataset_name,rows_max_bytes,expected",
    [
        # with rows_max_bytes > response size, the response is not truncated
        ("audio", 337 + SOME_BYTES, "complete"),
        ("image", 319 + SOME_BYTES, "complete"),
        ("images_list", 455 + SOME_BYTES, "complete"),
        ("audios_list", 447 + SOME_BYTES, "complete"),
        ("images_sequence", 484 + SOME_BYTES, "complete"),
        ("audios_sequence", 476 + SOME_BYTES, "complete"),
        ("dict_of_audios_and_images", 797 + SOME_BYTES, "complete"),
        # with rows_max_bytes < response size, the response is:
        # - not truncated for top-level Audio and Image features
        # - truncated for nested Audio and Image features
        ("audio", 337 - SOME_BYTES, "complete"),
        ("image", 319 - SOME_BYTES, "complete"),
        ("images_list", 455 - SOME_BYTES, "truncated_cells"),
        ("audios_list", 447 - SOME_BYTES, "truncated_cells"),
        ("images_sequence", 484 - SOME_BYTES, "truncated_cells"),
        ("audios_sequence", 476 - SOME_BYTES, "truncated_cells"),
        ("dict_of_audios_and_images", 797 - SOME_BYTES, "truncated_cells"),
    ],
)
def test_sign_urls_in_first_rows_in_place_with_truncated_cells(
    storage_client: StorageClient,
    datasets_fixtures: Mapping[str, DatasetFixture],
    dataset_name: str,
    rows_max_bytes: int,
    expected: Literal["truncated_cells", "complete"],
) -> None:
    dataset_fixture = datasets_fixtures[dataset_name]
    dataset = dataset_fixture.dataset

    first_rows = create_first_rows_response(
        dataset=dataset_name,
        revision=DEFAULT_REVISION,
        config=DEFAULT_CONFIG,
        split=DEFAULT_SPLIT,
        storage_client=storage_client,
        features=dataset.features,
        get_rows_content=get_dataset_rows_content(dataset=dataset),
        min_cell_bytes=DEFAULT_MIN_CELL_BYTES,
        rows_max_bytes=rows_max_bytes,
        rows_max_number=100,
        rows_min_number=DEFAULT_ROWS_MIN_NUMBER,
        columns_max_number=DEFAULT_COLUMNS_MAX_NUMBER,
    )

    url_signer = FakeUrlSigner()
    url_signer.sign_urls_in_first_rows_in_place(first_rows=first_rows)

    if expected == "complete":
        assert len(first_rows["rows"][0]["truncated_cells"]) == 0
        # ^ see test_rows.py
        assert url_signer.counter == dataset_fixture.expected_num_asset_urls
    else:
        assert len(first_rows["rows"][0]["truncated_cells"]) == 1
        # ^ see test_rows.py
        assert url_signer.counter == 0, first_rows
