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
ROWS_UVICORN_PORT = os.environ.get("ROWS_UVICORN_PORT", "8082")
SEARCH_UVICORN_PORT = os.environ.get("SEARCH_UVICORN_PORT", "8083")
ADMIN_TOKEN = os.environ.get("PARQUET_AND_INFO_COMMITTER_HF_TOKEN", "")
INTERVAL = 1
MAX_DURATION = 10 * 60
URL = f"http://localhost:{PORT_REVERSE_PROXY}"
ADMIN_URL = f"http://localhost:{ADMIN_UVICORN_PORT}"
API_URL = f"http://localhost:{API_UVICORN_PORT}"
ROWS_URL = f"http://localhost:{ROWS_UVICORN_PORT}"
SEARCH_URL = f"http://localhost:{SEARCH_UVICORN_PORT}"

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


def get_default_config_split() -> Tuple[str, str]:
    config = "default"
    split = "train"
    return config, split


def log(response: Response, url: str = URL, relative_url: Optional[str] = None, dataset: Optional[str] = None) -> str:
    if relative_url is not None:
        try:
            extra_response = get(
                f"/admin/cache-reports{relative_url}", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}, url=url
            )
            if extra_response.status_code == 200:
                extra = f"content of cache_reports: {extra_response.text}"
            else:
                extra = f"cannot get content of cache_reports: {extra_response.status_code} - {extra_response.text}"
        except Exception as e:
            extra = f"cannot get content of cache_reports - {e}"
        extra = f"\n{extra}"
    elif dataset is not None:
        try:
            extra_response = get(
                f"/admin/dataset-state?dataset={dataset}", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}, url=url
            )
            if extra_response.status_code == 200:
                extra = f"content of dataset-state: {extra_response.text}"
            else:
                extra = f"cannot get content of dataset-state: {extra_response.status_code} - {extra_response.text}"
        except Exception as e:
            extra = f"cannot get content of dataset-state - {e}"
        extra = f"\n{extra}"
    return (
        f"{dataset=} - {relative_url=} - {response.status_code} - {response.headers} - {response.text} - {url}{extra}"
    )


def poll_until_ready_and_assert(
    relative_url: str,
    expected_status_code: int,
    expected_error_code: Optional[str],
    headers: Optional[Headers] = None,
    url: str = URL,
    check_x_revision: bool = False,
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
        should_retry = response.headers.get("X-Error-Code") in ["ResponseNotReady", "ResponseAlreadyComputedError"]
    if retries == 0 or response is None:
        raise RuntimeError("Poll timeout")
    assert response.status_code == expected_status_code, log(response, url, relative_url)
    assert response.headers.get("X-Error-Code") == expected_error_code, log(response, url, relative_url)
    if check_x_revision:
        assert response.headers.get("X-Revision") is not None, log(response, url, relative_url)
        assert len(str(response.headers.get("X-Revision"))) == 40, log(response, url, relative_url)
    return response


# explicit re-export
__all__ = ["Response"]
