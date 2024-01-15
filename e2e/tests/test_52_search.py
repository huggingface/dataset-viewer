# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from .utils import get_default_config_split, poll_until_ready_and_assert


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
    assert "rows" in content, search_response
    assert "features" in content, search_response
    assert "num_rows_total" in content, search_response
    assert "num_rows_per_page" in content, search_response
    rows = content["rows"]
    features = content["features"]
    num_rows_total = content["num_rows_total"]
    num_rows_per_page = content["num_rows_per_page"]
    assert isinstance(rows, list), rows
    assert isinstance(features, list), features
    assert num_rows_total == 3
    assert num_rows_per_page == 100
    assert rows[0] == {
        "row_idx": 2,
        "row": {
            "col_1": "We count thirty Rebel ships, Lord Vader.",
            "col_2": 2,
            "col_3": 2.0,
            "col_4": "A",
            "col_5": True,
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
