# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Iterator
from typing import Literal, Optional

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

# TODO: test disabled repo (no way of changing that from the API)


def get_auth_headers(auth_type: str) -> dict[str, str]:
    return (
        {"Authorization": f"Bearer {NORMAL_USER_TOKEN}"}
        if auth_type == "token"
        else {"Cookie": f"token={NORMAL_USER_COOKIE}"}
        if auth_type == "cookie"
        else {}
    )


def poll_parquet_until_ready_and_assert(
    dataset: str,
    expected_status_code: Optional[int] = 200,
    expected_error_code: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
) -> None:
    # /parquet being successful means that:
    # - the dataset is supported
    # - both tokens worked: hf_token to read the datasets, and the parquet_converter token to write the parquet files
    response = poll_until_ready_and_assert(
        relative_url=f"/parquet?dataset={dataset}&config=default",
        expected_status_code=expected_status_code,
        expected_error_code=expected_error_code,
        headers=headers,
        check_x_revision=False,
        dataset=dataset,
    )
    if expected_status_code == 200:
        body = response.json()
        assert "parquet_files" in body, body
        assert len(body["parquet_files"]) == 1, body


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
    poll_parquet_until_ready_and_assert(
        dataset=normal_user_public_dataset,
        headers=get_auth_headers(auth_type),
        expected_status_code=expected_status_code,
        expected_error_code=expected_error_code,
    )


@pytest.fixture(scope="module")
def normal_user_gated_dataset(csv_path: str) -> Iterator[str]:
    with tmp_dataset(
        namespace=NORMAL_USER, token=NORMAL_USER_TOKEN, files={"data.csv": csv_path}, repo_settings={"gated": "auto"}
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
    poll_parquet_until_ready_and_assert(
        dataset=normal_user_gated_dataset,
        headers=get_auth_headers(auth_type),
        expected_status_code=expected_status_code,
        expected_error_code=expected_error_code,
    )


@pytest.fixture(scope="module")
def normal_user_private_dataset(csv_path: str) -> Iterator[str]:
    with tmp_dataset(
        namespace=NORMAL_USER, token=NORMAL_USER_TOKEN, files={"data.csv": csv_path}, repo_settings={"private": True}
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
    poll_parquet_until_ready_and_assert(
        dataset=normal_user_private_dataset,
        headers=get_auth_headers(auth_type),
        expected_status_code=expected_status_code,
        expected_error_code=expected_error_code,
    )


def test_normal_org_private(csv_path: str) -> None:
    with tmp_dataset(
        namespace=NORMAL_ORG, token=NORMAL_USER_TOKEN, files={"data.csv": csv_path}, repo_settings={"private": True}
    ) as dataset:
        poll_parquet_until_ready_and_assert(
            dataset=dataset,
            headers=get_auth_headers("token"),
            expected_status_code=404,
            expected_error_code="ResponseNotFound",
        )


def test_pro_user_private(csv_path: str) -> None:
    with tmp_dataset(
        namespace=PRO_USER, token=PRO_USER_TOKEN, files={"data.csv": csv_path}, repo_settings={"private": True}
    ) as dataset:
        poll_parquet_until_ready_and_assert(
            dataset=dataset,
            headers={"Authorization": f"Bearer {PRO_USER_TOKEN}"},
        )


def test_enterprise_user_private(csv_path: str) -> None:
    with tmp_dataset(
        namespace=ENTERPRISE_USER,
        token=ENTERPRISE_USER_TOKEN,
        files={"data.csv": csv_path},
        repo_settings={"private": True},
    ) as dataset:
        poll_parquet_until_ready_and_assert(
            dataset=dataset,
            headers={"Authorization": f"Bearer {ENTERPRISE_USER_TOKEN}"},
            expected_status_code=404,
            expected_error_code="ResponseNotFound",
        )


def test_enterprise_org_private(csv_path: str) -> None:
    with tmp_dataset(
        namespace=ENTERPRISE_ORG,
        token=ENTERPRISE_USER_TOKEN,
        files={"data.csv": csv_path},
        repo_settings={"private": True},
    ) as dataset:
        poll_parquet_until_ready_and_assert(
            dataset=dataset,
            headers={"Authorization": f"Bearer {ENTERPRISE_USER_TOKEN}"},
        )


def test_normal_user_private_gated(csv_path: str) -> None:
    with tmp_dataset(
        namespace=NORMAL_USER,
        token=NORMAL_USER_TOKEN,
        files={"data.csv": csv_path},
        repo_settings={"private": True, "gated": "auto"},
    ) as dataset:
        poll_parquet_until_ready_and_assert(
            dataset=dataset,
            headers={"Authorization": f"Bearer {NORMAL_USER_TOKEN}"},
            expected_status_code=404,
            expected_error_code="ResponseNotFound",
        )


def test_pro_user_private_gated(csv_path: str) -> None:
    with tmp_dataset(
        namespace=PRO_USER,
        token=PRO_USER_TOKEN,
        files={"data.csv": csv_path},
        repo_settings={"private": True, "gated": "auto"},
    ) as dataset:
        poll_parquet_until_ready_and_assert(
            dataset=dataset,
            headers={"Authorization": f"Bearer {PRO_USER_TOKEN}"},
        )


def test_normal_user_blocked(csv_path: str) -> None:
    with tmp_dataset(
        namespace=NORMAL_USER,
        token=NORMAL_USER_TOKEN,
        files={"data.csv": csv_path},
        dataset_prefix="blocked-",
        # ^ should be caught by COMMON_BLOCKED_DATASETS := "__DUMMY_DATASETS_SERVER_USER__/blocked-*"
    ) as dataset:
        poll_parquet_until_ready_and_assert(
            dataset=dataset, expected_status_code=404, expected_error_code="ResponseNotFound"
        )


@pytest.fixture
def disabled_viewer_readme_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "README.md")
    with open(path, "w", newline="") as f:
        f.writelines(
            [
                "---\n",
                "viewer: false\n",
                "---\n",
                "\n",
                "# Dataset\n",
            ]
        )
    return path


def test_normal_user_disabled_viewer(csv_path: str, disabled_viewer_readme_path: str) -> None:
    with tmp_dataset(
        namespace=NORMAL_USER,
        token=NORMAL_USER_TOKEN,
        files={"data.csv": csv_path, "README.md": disabled_viewer_readme_path},
    ) as dataset:
        poll_parquet_until_ready_and_assert(
            dataset=dataset, expected_status_code=404, expected_error_code="ResponseNotFound"
        )


def test_normal_user_disabled_discussions(csv_path: str) -> None:
    # it should have no effect, but we saw a strange bug once where disabling discussions prevented the dataset from being read
    with tmp_dataset(
        namespace=NORMAL_USER,
        token=NORMAL_USER_TOKEN,
        files={"data.csv": csv_path},
        repo_settings={"discussionsDisabled": True},
    ) as dataset:
        poll_parquet_until_ready_and_assert(dataset=dataset)


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
