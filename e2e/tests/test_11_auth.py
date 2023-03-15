# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

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
    endpoints = [
        f"/config-names?dataset={dataset}",
        f"/splits?dataset={dataset}",
        f"/first-rows?dataset={dataset}&config={config}&split={split}",
        f"/parquet-and-dataset-info?dataset={dataset}",
        f"/parquet?dataset={dataset}",
        f"/dataset-info?dataset={dataset}",
        f"/size?dataset={dataset}",
    ]
    for endpoint in endpoints:
        poll_until_ready_and_assert(
            relative_url=endpoint,
            expected_status_code=expected_status_code,
            expected_error_code=expected_error_code,
            headers=headers,
        )

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
        rows = content["rows"]
        assert isinstance(rows, list), rows
        assert len(rows) == 3, rows
        assert rows[0] == {"col_1": 1, "col_2": 1, "col_3": 1.0}, rows[0]
