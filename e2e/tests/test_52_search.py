# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from .fixtures.hub import AuthHeaders, AuthType
from .utils import get_default_config_split, poll_until_ready_and_assert


def test_search_endpoint(
    auth_headers: AuthHeaders,
    hf_public_dataset_repo_csv_data: str,
) -> None:
    auth: AuthType = "none"
    expected_status_code: int = 200
    expected_error_code = None
    # TODO: add dataset with various splits, or various configs
    dataset, config, split = get_default_config_split(hf_public_dataset_repo_csv_data)
    headers = auth_headers[auth]
    # ensure the /search endpoint works as well
    offset = 1
    length = 2
    query = "Lord Vader"
    search_response = poll_until_ready_and_assert(
        relative_url=(
            f"/search?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}&query={query}"
        ),
        expected_status_code=expected_status_code,
        expected_error_code=expected_error_code,
        headers=headers,
        check_x_revision=True,
    )
    if not expected_error_code:
        content = search_response.json()
        assert "rows" in content, search_response
        assert "features" in content, search_response
        assert "num_total_rows" in content, search_response
        rows = content["rows"]
        features = content["features"]
        num_total_rows = content["num_total_rows"]
        assert isinstance(rows, list), rows
        assert isinstance(features, list), features
        assert num_total_rows == 3
        assert rows[0] == {
            "row_idx": 2,
            "row": {"col_1": "We count thirty Rebel ships, Lord Vader.", "col_2": 2, "col_3": 2.0},
            "truncated_cells": [],
        }, rows[0]
        assert rows[1] == {
            "row_idx": 3,
            "row": {
                "col_1": "The wingman spots the pirateship coming at him and warns the Dark Lord",
                "col_2": 3,
                "col_3": 3.0,
            },
            "truncated_cells": [],
        }, rows[1]
        assert features == [
            {"feature_idx": 0, "name": "col_1", "type": {"dtype": "string", "_type": "Value"}},
            {"feature_idx": 1, "name": "col_2", "type": {"dtype": "int64", "_type": "Value"}},
            {"feature_idx": 2, "name": "col_3", "type": {"dtype": "float64", "_type": "Value"}},
        ], features
