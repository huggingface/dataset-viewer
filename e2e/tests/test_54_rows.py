# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from .utils import get_default_config_split, poll, poll_until_ready_and_assert


def test_rows_endpoint(normal_user_public_dataset: str) -> None:
    # TODO: add dataset with various splits, or various configs
    dataset = normal_user_public_dataset
    config, split = get_default_config_split()
    # ensure the /rows endpoint works as well
    offset = 1
    length = 10
    rows_response = poll_until_ready_and_assert(
        relative_url=f"/rows?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}",
        check_x_revision=True,
        dataset=dataset,
    )
    content = rows_response.json()
    assert "rows" in content, rows_response
    assert "features" in content, rows_response
    rows = content["rows"]
    features = content["features"]
    assert isinstance(rows, list), rows
    assert isinstance(features, list), features
    assert len(rows) == 3, rows
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
    assert features == [
        {"feature_idx": 0, "name": "col_1", "type": {"dtype": "string", "_type": "Value"}},
        {"feature_idx": 1, "name": "col_2", "type": {"dtype": "int64", "_type": "Value"}},
        {"feature_idx": 2, "name": "col_3", "type": {"dtype": "float64", "_type": "Value"}},
        {"feature_idx": 3, "name": "col_4", "type": {"dtype": "string", "_type": "Value"}},
        {"feature_idx": 4, "name": "col_5", "type": {"dtype": "bool", "_type": "Value"}},
    ], features


def test_rows_images_endpoint(normal_user_images_public_dataset: str) -> None:
    dataset = normal_user_images_public_dataset
    config, split = get_default_config_split()
    # ensure the /rows endpoint works as well
    offset = 1
    length = 10
    rows_response = poll_until_ready_and_assert(
        relative_url=f"/rows?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}",
        check_x_revision=True,
        dataset=dataset,
    )
    content = rows_response.json()
    # ensure the URL is signed
    url = content["rows"][0]["row"]["image"]["src"]
    assert "image.jpg?Expires=" in url, url
    assert "&Signature=" in url, url
    assert "&Key-Pair-Id=" in url, url
    # ensure the URL is valid
    response = poll(url)
    assert response.status_code == 200, response


def test_rows_audios_endpoint(normal_user_audios_public_dataset: str) -> None:
    dataset = normal_user_audios_public_dataset
    config, split = get_default_config_split()
    # ensure the /rows endpoint works as well
    offset = 1
    length = 10
    rows_response = poll_until_ready_and_assert(
        relative_url=f"/rows?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}",
        check_x_revision=True,
        dataset=dataset,
    )
    content = rows_response.json()
    # ensure the URL is signed
    url = content["rows"][0]["row"]["audio"][0]["src"]
    assert "audio.wav?Expires=" in url, url
    assert "&Signature=" in url, url
    assert "&Key-Pair-Id=" in url, url
    # ensure the URL is valid
    response = poll(url)
    assert response.status_code == 200, response
