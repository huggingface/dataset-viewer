# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Literal

import pytest

from .fixtures.hub import AuthHeaders, AuthType, DatasetRepos, DatasetReposType
from .utils import get_default_config_split, poll_until_ready_and_assert


@pytest.mark.parametrize(
    "type,auth,expected_status_code,expected_error_code",
    [
        ("public", "none", 200, None),
        ("public", "token", 200, None),
        ("public", "cookie", 200, None),
        ("gated", "none", 401, "ExternalUnauthenticatedError"),
        ("gated", "token", 200, None),
        ("gated", "cookie", 200, None),
        ("private", "none", 401, "ExternalUnauthenticatedError"),
        ("private", "token", 404, "ResponseNotFound"),
        ("private", "cookie", 404, "ResponseNotFound"),
    ],
)
def test_auth_e2e(
    auth_headers: AuthHeaders,
    hf_dataset_repos_csv_data: DatasetRepos,
    type: DatasetReposType,
    auth: AuthType,
    expected_status_code: int,
    expected_error_code: str,
) -> None:
    # TODO: add dataset with various splits, or various configs
    dataset, config, split = get_default_config_split(hf_dataset_repos_csv_data[type])
    headers = auth_headers[auth]

    # asking for the dataset will launch the jobs, without the need of a webhook
    poll_until_ready_and_assert(
        relative_url=f"/config-names?dataset={dataset}",
        expected_status_code=expected_status_code,
        expected_error_code=expected_error_code,
        headers=headers,
    )


@pytest.mark.parametrize(
    "endpoint,input_type",
    [
        ("/config-names", "dataset"),
        ("/splits", "dataset"),
        ("/splits", "config"),
        ("/first-rows", "split"),
        ("/parquet-and-dataset-info", "config"),
        ("/parquet", "dataset"),
        ("/parquet", "config"),
        ("/dataset-info", "dataset"),
        ("/dataset-info", "config"),
        ("/size", "dataset"),
        ("/size", "config"),
        ("/is-valid", "dataset"),
        ("/valid", "all"),
    ],
)
def test_endpoint(
    auth_headers: AuthHeaders,
    hf_public_dataset_repo_csv_data: str,
    endpoint: str,
    input_type: Literal["all", "dataset", "config", "split"],
) -> None:
    auth: AuthType = "none"
    expected_status_code: int = 200
    expected_error_code = None
    # TODO: add dataset with various splits, or various configs
    dataset, config, split = get_default_config_split(hf_public_dataset_repo_csv_data)
    headers = auth_headers[auth]

    # asking for the dataset will launch the jobs, without the need of a webhook
    relative_url = endpoint
    if input_type != "all":
        relative_url += f"?dataset={dataset}"
        if input_type != "dataset":
            relative_url += f"&config={config}"
            if input_type != "config":
                relative_url += f"&split={split}"

    poll_until_ready_and_assert(
        relative_url=relative_url,
        expected_status_code=expected_status_code,
        expected_error_code=expected_error_code,
        headers=headers,
    )


def test_rows_endpoint(
    auth_headers: AuthHeaders,
    hf_public_dataset_repo_csv_data: str,
) -> None:
    auth: AuthType = "none"
    expected_status_code: int = 200
    expected_error_code = None
    # TODO: add dataset with various splits, or various configs
    dataset, config, split = get_default_config_split(hf_public_dataset_repo_csv_data)
    headers = auth_headers[auth]
    # ensure the /rows endpoint works as well
    offset = 1
    limit = 10
    rows_response = poll_until_ready_and_assert(
        relative_url=f"/rows?dataset={dataset}&config={config}&split={split}&offset={offset}&limit={limit}",
        expected_status_code=expected_status_code,
        expected_error_code=expected_error_code,
        headers=headers,
    )
    if not expected_error_code:
        content = rows_response.json()
        assert "rows" in content, rows_response
        assert "features" in content, rows_response
        rows = content["rows"]
        features = content["features"]
        assert isinstance(rows, list), rows
        assert isinstance(features, list), features
        assert len(rows) == 3, rows
        assert rows[0] == {"row_idx": 1, "row": {"col_1": 1, "col_2": 1, "col_3": 1.0}, "truncated_cells": []}, rows[0]
        assert features == [
            {"feature_idx": 0, "name": "col_1", "type": {"dtype": "int64", "_type": "Value"}},
            {"feature_idx": 1, "name": "col_2", "type": {"dtype": "int64", "_type": "Value"}},
            {"feature_idx": 2, "name": "col_3", "type": {"dtype": "float64", "_type": "Value"}},
        ], features
