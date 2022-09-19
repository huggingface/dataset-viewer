# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
from requests import Response

PORT_REVERSE_PROXY = os.environ.get("PORT_REVERSE_PROXY", "8000")
ROWS_MAX_NUMBER = int(os.environ.get("ROWS_MAX_NUMBER", 100))
INTERVAL = 1
MAX_DURATION = 10 * 60
URL = f"http://localhost:{PORT_REVERSE_PROXY}"

Headers = Dict[str, str]


def get(relative_url: str, headers: Headers = None) -> Response:
    if headers is None:
        headers = {}
    return requests.get(f"{URL}{relative_url}", headers=headers)


def post(relative_url: str, json: Optional[Any] = None, headers: Headers = None) -> Response:
    if headers is None:
        headers = {}
    return requests.post(f"{URL}{relative_url}", json=json, headers=headers)


def poll(
    relative_url: str, error_field: Optional[str] = None, expected_code: Optional[int] = 200, headers: Headers = None
) -> Response:
    if headers is None:
        headers = {}
    interval = INTERVAL
    timeout = MAX_DURATION
    retries = timeout // interval
    should_retry = True
    response = None
    while retries > 0 and should_retry:
        retries -= 1
        time.sleep(interval)
        response = get(relative_url, headers)
        if error_field is not None:
            # currently, when the dataset is being processed, the error message contains "Retry later"
            try:
                should_retry = "retry later" in response.json()[error_field].lower()
            except Exception:
                should_retry = False
        else:
            # just retry if the response is not the expected code
            should_retry = response.status_code != expected_code
    if response is None:
        raise RuntimeError("no request has been done")
    return response


def post_refresh(dataset: str, headers: Headers = None) -> Response:
    if headers is None:
        headers = {}
    return post("/webhook", json={"update": f"datasets/{dataset}"}, headers=headers)


def poll_splits(dataset: str, headers: Headers = None) -> Response:
    return poll(f"/splits?dataset={dataset}", error_field="error", headers=headers)


def poll_first_rows(dataset: str, config: str, split: str, headers: Headers = None) -> Response:
    return poll(f"/first-rows?dataset={dataset}&config={config}&split={split}", error_field="error", headers=headers)


def refresh_poll_splits(dataset: str, headers: Headers = None) -> Response:
    # ask for the dataset to be refreshed
    response = post_refresh(dataset, headers=headers)
    assert response.status_code == 200, f"{response.status_code} - {response.text}"

    # poll the /splits endpoint until we get something else than "The dataset is being processed. Retry later."
    return poll_splits(dataset, headers=headers)


def refresh_poll_splits_first_rows(
    dataset: str, config: str, split: str, headers: Headers = None
) -> Tuple[Response, Response]:
    response_splits = refresh_poll_splits(dataset, headers=headers)
    assert response_splits.status_code == 200, f"{response_splits.status_code} - {response_splits.text}"

    response_rows = poll_first_rows(dataset, config, split, headers=headers)

    return response_splits, response_rows


def get_openapi_body_example(path, status, example_name):
    root = Path(__file__).resolve().parent.parent.parent
    openapi_filename = root / "chart" / "static-files" / "openapi.json"
    with open(openapi_filename) as json_file:
        openapi = json.load(json_file)
    return openapi["paths"][path]["get"]["responses"][str(status)]["content"]["application/json"]["examples"][
        example_name
    ]["value"]


def get_default_config_split(dataset: str) -> Tuple[str, str, str]:
    config = dataset.replace("/", "--")
    split = "train"
    return dataset, config, split


# explicit re-export
__all__ = ["Response"]
