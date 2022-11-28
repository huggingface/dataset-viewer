# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import json
from typing import Any

import pytest
import requests

from .utils import (
    URL,
    get,
    get_openapi_body_example,
    poll,
    poll_first_rows,
    poll_splits,
    post_refresh,
)


def prepare_json(response: requests.Response) -> Any:
    return json.loads(response.text.replace(URL, "https://datasets-server.huggingface.co"))


@pytest.mark.parametrize(
    "status,name,dataset,config,split,error_code",
    [
        (
            401,
            "inexistent-dataset",
            "severo/inexistent-dataset",
            "plain_text",
            "train",
            "ExternalUnauthenticatedError",
        ),
        (422, "missing-dataset", None, "plain_text", "train", "MissingRequiredParameter"),
        (422, "missing-config", "imdb", None, "train", "MissingRequiredParameter"),
        (422, "missing-split", "imdb", "plain_text", None, "MissingRequiredParameter"),
        (422, "empty-dataset", "", "plain_text", "train", "MissingRequiredParameter"),
        (422, "empty-config", "imdb", "", "train", "MissingRequiredParameter"),
        (422, "empty-split", "imdb", "plain_text", "", "MissingRequiredParameter"),
    ],
)
def test_first_rows(status: int, name: str, dataset: str, config: str, split: str, error_code: str):
    body = get_openapi_body_example("/first-rows", status, name)

    # the logic here is a bit convoluted, because we have no way to refresh a split, we have to refresh the whole
    # dataset and depend on the result of /splits
    if name.startswith("empty-"):
        r_rows = poll(f"/first-rows?dataset={dataset}&config={config}&split={split}", error_field="error")
    elif name.startswith("missing-"):
        d = f"dataset={dataset}" if dataset is not None else ""
        c = f"config={config}" if config is not None else ""
        s = f"split={split}" if split is not None else ""
        params = "&".join([d, c, s])
        r_rows = poll(f"/first-rows?{params}", error_field="error")
    else:
        post_refresh(dataset)
        poll_splits(dataset)
        if name == "not-ready":
            # poll the endpoint before the worker had the chance to process it
            r_rows = get(f"/first-rows?dataset={dataset}&config={config}&split={split}")
        else:
            r_rows = poll_first_rows(dataset, config, split)

    assert r_rows.status_code == status, f"{r_rows.status_code} - {r_rows.text}"
    assert prepare_json(r_rows) == body, r_rows.text
    if error_code is not None:
        assert r_rows.headers["X-Error-Code"] == error_code, r_rows.headers["X-Error-Code"]
    else:
        assert "X-Error-Code" not in r_rows.headers, r_rows.headers["X-Error-Code"]
