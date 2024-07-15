# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from collections.abc import Mapping
from typing import Literal

import pandas as pd
import pytest
from datasets import Dataset

from libcommon.constants import MAX_NUM_ROWS_PER_PAGE
from libcommon.exceptions import TooBigContentError
from libcommon.storage_client import StorageClient
from libcommon.viewer_utils.rows import create_first_rows_response

from ..constants import (
    DATASETS_NAMES,
    DEFAULT_COLUMN_NAME,
    DEFAULT_COLUMNS_MAX_NUMBER,
    DEFAULT_CONFIG,
    DEFAULT_MIN_CELL_BYTES,
    DEFAULT_REVISION,
    DEFAULT_ROWS_MAX_BYTES,
    DEFAULT_ROWS_MAX_NUMBER,
    DEFAULT_ROWS_MIN_NUMBER,
    DEFAULT_SPLIT,
    SOME_BYTES,
)
from ..types import DatasetFixture
from ..utils import get_dataset_rows_content


@pytest.mark.parametrize("dataset_name", DATASETS_NAMES)
def test_create_first_rows_response(
    storage_client_with_url_preparator: StorageClient,
    datasets_fixtures: Mapping[str, DatasetFixture],
    dataset_name: str,
) -> None:
    dataset_fixture = datasets_fixtures[dataset_name]
    dataset = dataset_fixture.dataset

    response = create_first_rows_response(
        dataset=dataset_name,
        revision=DEFAULT_REVISION,
        config=DEFAULT_CONFIG,
        split=DEFAULT_SPLIT,
        storage_client=storage_client_with_url_preparator,
        features=dataset.features,
        get_rows_content=get_dataset_rows_content(dataset=dataset),
        min_cell_bytes=DEFAULT_MIN_CELL_BYTES,
        rows_max_bytes=DEFAULT_ROWS_MAX_BYTES,
        rows_max_number=DEFAULT_ROWS_MAX_NUMBER,
        rows_min_number=DEFAULT_ROWS_MIN_NUMBER,
        columns_max_number=DEFAULT_COLUMNS_MAX_NUMBER,
    )
    assert not response["truncated"]
    assert response["features"][0]["type"] == dataset_fixture.expected_feature_type
    assert response["rows"][0]["row"] == {DEFAULT_COLUMN_NAME: dataset_fixture.expected_cell}
    assert response["rows"][0]["truncated_cells"] == []


NUM_ROWS = 15


@pytest.mark.parametrize(
    "rows_max_bytes,rows_max_number,truncated",
    [
        (1_000, NUM_ROWS + 5, True),  # truncated because of rows_max_bytes
        (10_000_000_000, NUM_ROWS - 5, True),  # truncated because of rows_max_number
        (10_000_000_000, NUM_ROWS + 5, False),  # not truncated
    ],
)
def test_create_first_rows_response_truncated(
    storage_client: StorageClient,
    rows_max_bytes: int,
    rows_max_number: int,
    truncated: bool,
) -> None:
    CELL_SIZE = 1_234
    dataset = Dataset.from_pandas(
        pd.DataFrame(
            ["a" * CELL_SIZE for _ in range(NUM_ROWS)],
            dtype=pd.StringDtype(storage="python"),
        )
    )

    response = create_first_rows_response(
        dataset="dataset",
        revision=DEFAULT_REVISION,
        config=DEFAULT_CONFIG,
        split=DEFAULT_SPLIT,
        storage_client=storage_client,
        features=dataset.features,
        get_rows_content=get_dataset_rows_content(dataset=dataset),
        min_cell_bytes=DEFAULT_MIN_CELL_BYTES,
        rows_max_bytes=rows_max_bytes,
        rows_max_number=rows_max_number,
        rows_min_number=DEFAULT_ROWS_MIN_NUMBER,
        columns_max_number=DEFAULT_COLUMNS_MAX_NUMBER,
    )
    assert response["truncated"] == truncated


@pytest.mark.parametrize(
    "dataset_name,rows_max_bytes,expected",
    [
        # with rows_max_bytes > response size, the response is not truncated
        ("audio", 337 + SOME_BYTES, "complete"),
        ("image", 319 + SOME_BYTES, "complete"),
        ("urls", 268 + SOME_BYTES, "complete"),
        ("images_list", 455 + SOME_BYTES, "complete"),
        ("audios_list", 447 + SOME_BYTES, "complete"),
        ("images_sequence", 484 + SOME_BYTES, "complete"),
        ("audios_sequence", 476 + SOME_BYTES, "complete"),
        ("dict_of_audios_and_images", 797 + SOME_BYTES, "complete"),
        # with rows_max_bytes < response size, the response is:
        # - not truncated for top-level Audio and Image features and urls
        # - truncated for nested Audio and Image features
        ("audio", 337 - SOME_BYTES, "complete"),
        ("image", 319 - SOME_BYTES, "complete"),
        ("urls", 268 - SOME_BYTES, "complete"),
        ("images_list", 455 - SOME_BYTES, "truncated_cells"),
        ("audios_list", 447 - SOME_BYTES, "truncated_cells"),
        ("images_sequence", 484 - SOME_BYTES, "truncated_cells"),
        ("audios_sequence", 476 - SOME_BYTES, "truncated_cells"),
        ("dict_of_audios_and_images", 797 - SOME_BYTES, "truncated_cells"),
        # with rows_max_bytes <<< response size, a TooBigContentError exception is raised
        # (note that it should never happen if the correct set of parameters is chosen)
        ("audio", 10, "error"),
        ("image", 10, "error"),
        ("urls", 10, "error"),
        ("images_list", 10, "error"),
        ("audios_list", 10, "error"),
        ("images_sequence", 10, "error"),
        ("audios_sequence", 10, "error"),
        ("dict_of_audios_and_images", 10, "error"),
    ],
)
def test_create_first_rows_response_truncation_on_audio_or_image(
    storage_client_with_url_preparator: StorageClient,
    datasets_fixtures: Mapping[str, DatasetFixture],
    dataset_name: str,
    rows_max_bytes: int,
    expected: Literal["error", "truncated_cells", "complete"],
) -> None:
    dataset_fixture = datasets_fixtures[dataset_name]
    dataset = dataset_fixture.dataset

    if expected == "error":
        with pytest.raises(TooBigContentError):
            response = create_first_rows_response(
                dataset=dataset_name,
                revision=DEFAULT_REVISION,
                config=DEFAULT_CONFIG,
                split=DEFAULT_SPLIT,
                storage_client=storage_client_with_url_preparator,
                features=dataset.features,
                get_rows_content=get_dataset_rows_content(dataset=dataset),
                min_cell_bytes=DEFAULT_MIN_CELL_BYTES,
                rows_max_bytes=rows_max_bytes,
                rows_max_number=MAX_NUM_ROWS_PER_PAGE,
                rows_min_number=DEFAULT_ROWS_MIN_NUMBER,
                columns_max_number=DEFAULT_COLUMNS_MAX_NUMBER,
            )
            print(response)
    else:
        response = create_first_rows_response(
            dataset=dataset_name,
            revision=DEFAULT_REVISION,
            config=DEFAULT_CONFIG,
            split=DEFAULT_SPLIT,
            storage_client=storage_client_with_url_preparator,
            features=dataset.features,
            get_rows_content=get_dataset_rows_content(dataset=dataset),
            min_cell_bytes=DEFAULT_MIN_CELL_BYTES,
            rows_max_bytes=rows_max_bytes,
            rows_max_number=MAX_NUM_ROWS_PER_PAGE,
            rows_min_number=DEFAULT_ROWS_MIN_NUMBER,
            columns_max_number=DEFAULT_COLUMNS_MAX_NUMBER,
        )
        assert not response["truncated"]
        # ^ no rows have been deleted
        assert len(response["rows"][0]["truncated_cells"]) == (1 if (expected == "truncated_cells") else 0), response
