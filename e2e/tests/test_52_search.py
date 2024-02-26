# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from .utils import get_default_config_split, poll, poll_until_ready_and_assert


def test_search_endpoint(normal_user_public_dataset: str) -> None:
    # TODO: add dataset with various splits, or various configs
    dataset = normal_user_public_dataset
    config, split = get_default_config_split()
    # ensure the /search endpoint works as well
    offset = 1
    length = 2
    query = "Lord Vader"
    search_response = poll_until_ready_and_assert(
        relative_url=(
            f"/search?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}&query={query}"
        ),
        check_x_revision=True,
        dataset=dataset,
    )
    content = search_response.json()
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
        "row_idx": 2,
        "row": {
            "col_1": "We count thirty Rebel ships, Lord Vader.",
            "col_2": 2,
            "col_3": 2.0,
            "col_4": "A",
            "col_5": True,
            "col_6": [0, 1, 2, 3, None],
        },
        "truncated_cells": [],
    }, rows[0]
    assert rows[1] == {
        "row_idx": 3,
        "row": {
            "col_1": "The wingman spots the pirateship coming at him and warns the Dark Lord",
            "col_2": 3,
            "col_3": 3.0,
            "col_4": "B",
            "col_5": None,
            "col_6": None,
        },
        "truncated_cells": [],
    }, rows[1]
    assert features == [
        {"feature_idx": 0, "name": "col_1", "type": {"dtype": "string", "_type": "Value"}},
        {"feature_idx": 1, "name": "col_2", "type": {"dtype": "int64", "_type": "Value"}},
        {"feature_idx": 2, "name": "col_3", "type": {"dtype": "float64", "_type": "Value"}},
        {"feature_idx": 3, "name": "col_4", "type": {"dtype": "string", "_type": "Value"}},
        {"feature_idx": 4, "name": "col_5", "type": {"dtype": "bool", "_type": "Value"}},
        {"feature_idx": 5, "name": "col_5", "type": [{"dtype": "int", "_type": "Value"}]},
    ], features


def test_search_images_endpoint(normal_user_images_public_dataset: str) -> None:
    dataset = normal_user_images_public_dataset
    config, split = get_default_config_split()
    query = "yellow"
    rows_response = poll_until_ready_and_assert(
        relative_url=f"/search?dataset={dataset}&config={config}&split={split}&query={query}",
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


def test_search_audios_endpoint(normal_user_audios_public_dataset: str) -> None:
    dataset = normal_user_audios_public_dataset
    config, split = get_default_config_split()
    query = "small"
    rows_response = poll_until_ready_and_assert(
        relative_url=f"/search?dataset={dataset}&config={config}&split={split}&query={query}",
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
