# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from .fixtures.hub import AuthHeaders, AuthType, DatasetRepos, DatasetReposType
from .utils import (
    Response,
    get_default_config_split,
    poll_first_rows,
    poll_splits,
    post_refresh,
)


def log(response: Response, dataset: str) -> str:
    dataset, config, split = get_default_config_split(dataset)
    return f"{response.status_code} - {response.text} - {dataset} - {config} - {split}"


@pytest.mark.parametrize(
    "type,auth,webhook_status_code,response_status_code,error_code_splits,error_code_first_rows",
    [
        ("public", "none", 200, 200, None, None),
        ("public", "token", 200, 200, None, None),
        ("public", "cookie", 200, 200, None, None),
        # gated: webhook_status_code is 200 because the access is asked for the app token, not the user token
        # (which is not passed to the webhook request)
        ("gated", "none", 200, 401, "ExternalUnauthenticatedError", "ExternalUnauthenticatedError"),
        ("gated", "token", 200, 200, None, None),
        ("gated", "cookie", 200, 200, None, None),
        # private: webhook_status_code is 400 because the access is asked for the app token, which has no
        # access to the private datasets. As a consequence, no data in the cache
        ("private", "none", 400, 401, "ExternalUnauthenticatedError", "ExternalUnauthenticatedError"),
        ("private", "token", 400, 404, "ResponseNotFound", "ResponseNotFound"),
        ("private", "cookie", 400, 404, "ResponseNotFound", "ResponseNotFound"),
    ],
)
def test_split_public_auth(
    auth_headers: AuthHeaders,
    hf_dataset_repos_csv_data: DatasetRepos,
    type: DatasetReposType,
    auth: AuthType,
    webhook_status_code: int,
    response_status_code: int,
    error_code_splits: str,
    error_code_first_rows: str,
) -> None:
    dataset, config, split = get_default_config_split(hf_dataset_repos_csv_data[type])
    r_webhook = post_refresh(dataset)
    assert r_webhook.status_code == webhook_status_code, log(r_webhook, dataset)
    r_splits = poll_splits(dataset, headers=auth_headers[auth])
    assert r_splits.status_code == response_status_code, log(r_splits, dataset)
    assert r_splits.headers.get("X-Error-Code") == error_code_splits, log(r_splits, dataset)
    r_rows = poll_first_rows(dataset, config, split, headers=auth_headers[auth])
    assert r_rows.status_code == response_status_code, log(r_rows, dataset)
    assert r_rows.headers.get("X-Error-Code") == error_code_first_rows, log(r_rows, dataset)
