# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from .utils import get, get_openapi_body_example, poll, poll_splits, post_refresh


@pytest.mark.parametrize(
    "status,name,dataset,error_code",
    [
        # (200, "duorc", "duorc", None),
        # (200, "emotion", "emotion", None),
        (
            401,
            "inexistent-dataset",
            "severo/inexistent-dataset",
            "ExternalUnauthenticatedError",
        ),
        # (
        #     401,
        #     "gated-dataset",
        #     "severo/dummy_gated",
        #     "ExternalUnauthenticatedError",
        # ),
        # (
        #     401,
        #     "private-dataset",
        #     "severo/dummy_private",
        #     "ExternalUnauthenticatedError",
        # ),
        (422, "empty-parameter", "", "MissingRequiredParameter"),
        (422, "missing-parameter", None, "MissingRequiredParameter"),
        # (500, "SplitsNotFoundError", "natural_questions", "SplitsNamesError"),
        # (500, "FileNotFoundError", "akhaliq/test", "SplitsNamesError"),
        # (500, "not-ready", "severo/fix-401", "SplitsResponseNotReady"),
        # not tested: 'internal_error'
    ],
)
def test_splits_using_openapi(status: int, name: str, dataset: str, error_code: str) -> None:
    body = get_openapi_body_example("/splits", status, name)

    if name == "empty-parameter":
        r_splits = poll("/splits?dataset=", error_field="error")
    elif name == "missing-parameter":
        r_splits = poll("/splits", error_field="error")
    else:
        post_refresh(dataset)
        # poll the endpoint before the worker had the chance to process it
        r_splits = get(f"/splits?dataset={dataset}") if name == "not-ready" else poll_splits(dataset)

    assert r_splits.status_code == status, f"{r_splits.status_code} - {r_splits.text}"
    assert r_splits.json() == body, r_splits.text
    if error_code is not None:
        assert r_splits.headers["X-Error-Code"] == error_code, r_splits.headers["X-Error-Code"]
    else:
        assert "X-Error-Code" not in r_splits.headers, r_splits.headers["X-Error-Code"]


@pytest.mark.parametrize(
    "status,dataset,config,error_code",
    [
        # (200, "duorc", "SelfRC", None),
        (422, "inexistent-dataset-with-split", "my_config", "MissingRequiredParameter"),
    ],
)
def test_splits_with_config_using_openapi(status: int, dataset: str, config: str, error_code: str) -> None:
    r_splits = (
        poll(f"/splits?dataset={dataset}&config={config}&split=extra_param", error_field="error")
        if error_code
        else poll(f"/splits?dataset={dataset}&config={config}")
    )

    assert r_splits.status_code == status, f"{r_splits.status_code} - {r_splits.text}"

    if error_code is None:
        assert all(split["config"] == config for split in r_splits.json()["split_names"])
        # all splits must belong to the provided config

        assert "X-Error-Code" not in r_splits.headers, r_splits.headers["X-Error-Code"]
    else:
        assert r_splits.headers["X-Error-Code"] == error_code, r_splits.headers["X-Error-Code"]
