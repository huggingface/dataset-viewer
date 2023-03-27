# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import requests
from requests import Response

PORT_REVERSE_PROXY = os.environ.get("PORT_REVERSE_PROXY", "8000")
API_UVICORN_PORT = os.environ.get("API_UVICORN_PORT", "8080")
ADMIN_UVICORN_PORT = os.environ.get("ADMIN_UVICORN_PORT", "8081")
INTERVAL = 1
MAX_DURATION = 10 * 60
URL = f"http://localhost:{PORT_REVERSE_PROXY}"
ADMIN_URL = f"http://localhost:{ADMIN_UVICORN_PORT}"
API_URL = f"http://localhost:{API_UVICORN_PORT}"

Headers = Mapping[str, str]


def get(relative_url: str, headers: Optional[Headers] = None, url: str = URL) -> Response:
    if headers is None:
        headers = {}
    return requests.get(f"{url}{relative_url}", headers=headers)


def post(relative_url: str, json: Optional[Any] = None, headers: Optional[Headers] = None, url: str = URL) -> Response:
    if headers is None:
        headers = {}
    return requests.post(f"{url}{relative_url}", json=json, headers=headers)


def poll(
    relative_url: str,
    error_field: Optional[str] = None,
    expected_code: Optional[int] = 200,
    headers: Optional[Headers] = None,
    url: str = URL,
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
        response = get(relative_url=relative_url, headers=headers, url=url)
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


def post_refresh(dataset: str) -> Response:
    return post("/webhook", json={"event": "update", "repo": {"type": "dataset", "name": dataset}})


def poll_parquet(dataset: str, headers: Optional[Headers] = None) -> Response:
    return poll(f"/parquet?dataset={dataset}", error_field="error", headers=headers)


def poll_splits(dataset: str, headers: Optional[Headers] = None) -> Response:
    return poll(f"/splits?dataset={dataset}", error_field="error", headers=headers)


def poll_first_rows(dataset: str, config: str, split: str, headers: Optional[Headers] = None) -> Response:
    return poll(f"/first-rows?dataset={dataset}&config={config}&split={split}", error_field="error", headers=headers)


def get_openapi_body_example(path: str, status: int, example_name: str) -> Any:
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


def log(response: Response, url: str) -> str:
    return f"{response.status_code} - {response.headers} - {response.text} - {url}"


def poll_until_ready_and_assert(
    relative_url: str,
    expected_status_code: int,
    expected_error_code: Optional[str],
    headers: Optional[Headers] = None,
    url: str = URL,
) -> Any:
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
        response = get(relative_url=relative_url, headers=headers, url=url)
        print(response.headers.get("X-Error-Code"))
        should_retry = response.headers.get("X-Error-Code") == "ResponseNotReady"
    if retries == 0 or response is None:
        raise RuntimeError("Poll timeout")
    assert response.status_code == expected_status_code, log(response, url)
    assert response.headers.get("X-Error-Code") == expected_error_code, log(response, url)
    return response


# explicit re-export
__all__ = ["Response"]
