# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Iterator
from typing import Literal

import pytest

from .constants import (
    ENTERPRISE_ORG,
    ENTERPRISE_USER,
    ENTERPRISE_USER_TOKEN,
    NORMAL_ORG,
    NORMAL_USER,
    NORMAL_USER_COOKIE,
    NORMAL_USER_TOKEN,
    PRO_USER,
    PRO_USER_TOKEN,
)
from .utils import get_default_config_split, poll_until_ready_and_assert, tmp_dataset


def get_auth_headers(auth_type: str) -> dict[str, str]:
    return (
        {"Authorization": f"Bearer {NORMAL_USER_TOKEN}"}
        if auth_type == "token"
        else {"Cookie": f"token={NORMAL_USER_COOKIE}"}
        if auth_type == "cookie"
        else {}
    )


@pytest.mark.parametrize(
    "auth_type,expected_status_code,expected_error_code",
    [
        (None, 200, None),
        ("token", 200, None),
        ("cookie", 200, None),
    ],
)
def test_auth_public(
    normal_user_public_dataset: str,
    auth_type: str,
    expected_status_code: int,
    expected_error_code: str,
) -> None:
    # asking for the dataset will launch the jobs, without the need of a webhook
    poll_until_ready_and_assert(
        relative_url=f"/splits?dataset={normal_user_public_dataset}",
        expected_status_code=expected_status_code,
        expected_error_code=expected_error_code,
        headers=get_auth_headers(auth_type),
        check_x_revision=False,
        dataset=normal_user_public_dataset,
    )


@pytest.fixture(scope="module")
def normal_user_gated_dataset(csv_path: str) -> Iterator[str]:
    with tmp_dataset(
        namespace=NORMAL_USER,
        token=NORMAL_USER_TOKEN,
        private=False,
        gated="auto",
        csv_path=csv_path,
    ) as dataset:
        yield dataset


@pytest.mark.parametrize(
    "auth_type,expected_status_code,expected_error_code",
    [
        (None, 401, "ExternalUnauthenticatedError"),
        ("token", 200, None),
        ("cookie", 200, None),
    ],
)
def test_auth_gated(
    normal_user_gated_dataset: str,
    auth_type: str,
    expected_status_code: int,
    expected_error_code: str,
) -> None:
    # asking for the dataset will launch the jobs, without the need of a webhook
    poll_until_ready_and_assert(
        relative_url=f"/splits?dataset={normal_user_gated_dataset}",
        expected_status_code=expected_status_code,
        expected_error_code=expected_error_code,
        headers=get_auth_headers(auth_type),
        check_x_revision=False,
        dataset=normal_user_gated_dataset,
    )


@pytest.fixture(scope="module")
def normal_user_private_dataset(csv_path: str) -> Iterator[str]:
    with tmp_dataset(
        namespace=NORMAL_USER,
        token=NORMAL_USER_TOKEN,
        private=True,
        gated=None,
        csv_path=csv_path,
    ) as dataset:
        yield dataset


@pytest.mark.parametrize(
    "auth_type,expected_status_code,expected_error_code",
    [
        (None, 401, "ExternalUnauthenticatedError"),
        ("token", 404, "ResponseNotFound"),
        ("cookie", 404, "ResponseNotFound"),
    ],
)
def test_auth_private(
    normal_user_private_dataset: str,
    auth_type: str,
    expected_status_code: int,
    expected_error_code: str,
) -> None:
    # asking for the dataset will launch the jobs, without the need of a webhook
    poll_until_ready_and_assert(
        relative_url=f"/splits?dataset={normal_user_private_dataset}",
        expected_status_code=expected_status_code,
        expected_error_code=expected_error_code,
        headers=get_auth_headers(auth_type),
        check_x_revision=False,
        dataset=normal_user_private_dataset,
    )


# TODO: test private gated?
# TODO: test blocked datasets list
# TODO: test disabled repo
# TODO: test disabled viewer


def test_normal_org_private(csv_path: str) -> None:
    with tmp_dataset(
        namespace=NORMAL_ORG,
        token=NORMAL_USER_TOKEN,
        private=True,
        gated=None,
        csv_path=csv_path,
    ) as dataset:
        poll_until_ready_and_assert(
            relative_url=f"/splits?dataset={dataset}",
            expected_status_code=404,
            expected_error_code="ResponseNotFound",
            headers=get_auth_headers("token"),
            check_x_revision=False,
            dataset=dataset,
        )


def test_pro_user_private(csv_path: str) -> None:
    with tmp_dataset(
        namespace=PRO_USER,
        token=PRO_USER_TOKEN,
        private=True,
        gated=None,
        csv_path=csv_path,
    ) as dataset:
        poll_until_ready_and_assert(
            relative_url=f"/splits?dataset={dataset}",
            expected_status_code=200,
            expected_error_code=None,
            headers={"Authorization": f"Bearer {PRO_USER_TOKEN}"},
            check_x_revision=False,
            dataset=dataset,
        )


def test_enterprise_user_private(csv_path: str) -> None:
    with tmp_dataset(
        namespace=ENTERPRISE_USER,
        token=ENTERPRISE_USER_TOKEN,
        private=True,
        gated=None,
        csv_path=csv_path,
    ) as dataset:
        poll_until_ready_and_assert(
            relative_url=f"/splits?dataset={dataset}",
            expected_status_code=404,
            expected_error_code="ResponseNotFound",
            headers={"Authorization": f"Bearer {ENTERPRISE_USER_TOKEN}"},
            check_x_revision=False,
            dataset=dataset,
        )


def test_enterprise_org_private(csv_path: str) -> None:
    with tmp_dataset(
        namespace=ENTERPRISE_ORG,
        token=ENTERPRISE_USER_TOKEN,
        private=True,
        gated=None,
        csv_path=csv_path,
    ) as dataset:
        poll_until_ready_and_assert(
            relative_url=f"/splits?dataset={dataset}",
            expected_status_code=200,
            expected_error_code=None,
            headers={"Authorization": f"Bearer {ENTERPRISE_USER_TOKEN}"},
            check_x_revision=False,
            dataset=dataset,
        )


@pytest.mark.parametrize(
    "endpoint,input_type",
    [
        ("/splits", "dataset"),
        ("/splits", "config"),
        ("/first-rows", "split"),
        ("/parquet", "dataset"),
        ("/parquet", "config"),
        ("/info", "dataset"),
        ("/info", "config"),
        ("/size", "dataset"),
        ("/size", "config"),
        ("/is-valid", "dataset"),
        ("/statistics", "split"),
    ],
)
def test_endpoint(
    normal_user_public_dataset: str,
    endpoint: str,
    input_type: Literal["all", "dataset", "config", "split"],
) -> None:
    # TODO: add dataset with various splits, or various configs
    dataset = normal_user_public_dataset
    config, split = get_default_config_split()

    # asking for the dataset will launch the jobs, without the need of a webhook
    relative_url = endpoint
    if input_type != "all":
        relative_url += f"?dataset={dataset}"
        if input_type != "dataset":
            relative_url += f"&config={config}"
            if input_type != "config":
                relative_url += f"&split={split}"

    poll_until_ready_and_assert(relative_url=relative_url, check_x_revision=input_type != "all", dataset=dataset)


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
