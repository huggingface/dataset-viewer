# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import pytest

from .utils import get_default_config_split, poll, poll_until_ready_and_assert


def test_filter_endpoint(normal_user_public_dataset: str) -> None:
    # TODO: add dataset with various splits, or various configs
    dataset = normal_user_public_dataset
    config, split = get_default_config_split()
    offset = 1
    length = 2
    where = "col_4='B'"
    orderby = "col_2 DESC"
    filter_response = poll_until_ready_and_assert(
        relative_url=(
            f"/filter?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}&where={where}&orderby={orderby}"
        ),
        check_x_revision=True,
        dataset=dataset,
    )
    content = filter_response.json()
    assert sorted(content) == sorted(["rows", "features", "num_rows_total", "num_rows_per_page", "partial"])
    rows = content["rows"]
    features = content["features"]
    num_rows_total = content["num_rows_total"]
    num_rows_per_page = content["num_rows_per_page"]
    partial = content["partial"]
    assert isinstance(rows, list), rows
    assert isinstance(features, list), features
    assert num_rows_total == 3
    assert num_rows_per_page == 100
    assert partial is False
    assert rows[0] == {
        "row_idx": 1,
        "row": {
            "col_1": "Vader turns round and round in circles as his ship spins into space.",
            "col_2": 1,
            "col_3": 1.0,
            "col_4": "B",
            "col_5": False,
        },
        "truncated_cells": [],
    }, rows[0]
    assert rows[1] == {
        "row_idx": 0,
        "row": {
            "col_1": "There goes another one.",
            "col_2": 0,
            "col_3": 0.0,
            "col_4": "B",
            "col_5": True,
        },
        "truncated_cells": [],
    }, rows[1]
    assert features == [
        {"feature_idx": 0, "name": "col_1", "type": {"dtype": "string", "_type": "Value"}},
        {"feature_idx": 1, "name": "col_2", "type": {"dtype": "int64", "_type": "Value"}},
        {"feature_idx": 2, "name": "col_3", "type": {"dtype": "float64", "_type": "Value"}},
        {"feature_idx": 3, "name": "col_4", "type": {"dtype": "string", "_type": "Value"}},
        {"feature_idx": 4, "name": "col_5", "type": {"dtype": "bool", "_type": "Value"}},
    ], features


@pytest.mark.parametrize(
    "where,expected_num_rows",
    [
        ("", 4),
        ("col_2=3", 1),
        ("col_2<3", 3),
        ("col_2>3", 0),
        ("col_4='B'", 3),
        ("col_4<'B'", 1),
        ("col_4>='A'", 4),
        ("col_2<3 AND col_4='B'", 2),
        ("col_2<3 OR col_4='B'", 4),
    ],
)
def test_filter_endpoint_parameter_where(where: str, expected_num_rows: int, normal_user_public_dataset: str) -> None:
    dataset = normal_user_public_dataset
    config, split = get_default_config_split()
    relative_url = f"/filter?dataset={dataset}&config={config}&split={split}"
    if where:
        relative_url += f"&where={where}"
    response = poll_until_ready_and_assert(
        relative_url=relative_url,
        check_x_revision=True,
        dataset=dataset,
    )
    content = response.json()
    assert "rows" in content, response
    assert len(content["rows"]) == expected_num_rows


@pytest.mark.parametrize(
    "orderby, expected_first_row_idx",
    [
        ("", 1),
        ("col_4", 2),
        ("col_3 DESC", 3),
    ],
)
def test_filter_endpoint_parameter_orderby(
    orderby: str, expected_first_row_idx: int, normal_user_public_dataset: str
) -> None:
    dataset = normal_user_public_dataset
    config, split = get_default_config_split()
    where = "col_2>0"
    relative_url = f"/filter?dataset={dataset}&config={config}&split={split}&where={where}"
    if orderby:
        relative_url += f"&orderby={orderby}"
    response = poll_until_ready_and_assert(
        relative_url=relative_url,
        check_x_revision=True,
        dataset=dataset,
    )
    content = response.json()
    assert "rows" in content, response
    rows = content["rows"]
    assert rows[0]["row_idx"] == expected_first_row_idx, rows[0]


def test_filter_images_endpoint(normal_user_images_public_dataset: str) -> None:
    dataset = normal_user_images_public_dataset
    config, split = get_default_config_split()
    where = "rating=3"
    rows_response = poll_until_ready_and_assert(
        relative_url=f"/filter?dataset={dataset}&config={config}&split={split}&where={where}",
        dataset=dataset,
        should_retry_x_error_codes=["ResponseNotFound"],
        # ^ I had 404 errors without it. It should return something else at one point.
    )
    content = rows_response.json()
    # ensure the URL is signed
    url = content["rows"][0]["row"]["image"]["src"]
    assert "image.jpg?Expires=" in url, url
    assert "&Signature=" in url, url
    assert "&Key-Pair-Id=" in url, url
    # ensure the URL is valid
    response = poll(url, url="")
    assert response.status_code == 200, response


def test_filter_audios_endpoint(normal_user_audios_public_dataset: str) -> None:
    dataset = normal_user_audios_public_dataset
    config, split = get_default_config_split()
    where = "age=3"
    rows_response = poll_until_ready_and_assert(
        relative_url=f"/filter?dataset={dataset}&config={config}&split={split}&where={where}",
        dataset=dataset,
        should_retry_x_error_codes=["ResponseNotFound"],
        # ^ I had 404 errors without it. It should return something else at one point.
    )
    content = rows_response.json()
    # ensure the URL is signed
    url = content["rows"][0]["row"]["audio"][0]["src"]
    assert "audio.wav?Expires=" in url, url
    assert "&Signature=" in url, url
    assert "&Key-Pair-Id=" in url, url
    # ensure the URL is valid
    response = poll(url, url="")
    assert response.status_code == 200, response
