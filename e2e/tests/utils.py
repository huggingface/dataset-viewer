# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import json
import os
import re
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Optional, Union

import requests
from huggingface_hub.constants import REPO_TYPES, REPO_TYPES_URL_PREFIXES
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils._errors import hf_raise_for_status
from requests import Response

from .constants import CI_HUB_ENDPOINT

PORT_REVERSE_PROXY = os.environ.get("PORT_REVERSE_PROXY", "8000")
API_UVICORN_PORT = os.environ.get("API_UVICORN_PORT", "8080")
ADMIN_UVICORN_PORT = os.environ.get("ADMIN_UVICORN_PORT", "8081")
ROWS_UVICORN_PORT = os.environ.get("ROWS_UVICORN_PORT", "8082")
SEARCH_UVICORN_PORT = os.environ.get("SEARCH_UVICORN_PORT", "8083")
WORKER_UVICORN_PORT = os.environ.get("WORKER_UVICORN_PORT", "8086")
E2E_ADMIN_USER_TOKEN = os.environ.get("E2E_ADMIN_USER_TOKEN", "")
INTERVAL = 1
MAX_DURATION = 10 * 60
URL = f"http://localhost:{PORT_REVERSE_PROXY}"
ADMIN_URL = f"http://localhost:{ADMIN_UVICORN_PORT}"
API_URL = f"http://localhost:{API_UVICORN_PORT}"
ROWS_URL = f"http://localhost:{ROWS_UVICORN_PORT}"
SEARCH_URL = f"http://localhost:{SEARCH_UVICORN_PORT}"
WORKER_URL = f"http://localhost:{WORKER_UVICORN_PORT}"

Headers = Mapping[str, str]


def get(relative_url: str, headers: Optional[Headers] = None, url: str = URL) -> Response:
    return requests.get(f"{url}{relative_url}", headers=headers)


def post(relative_url: str, json: Optional[Any] = None, headers: Optional[Headers] = None, url: str = URL) -> Response:
    return requests.post(f"{url}{relative_url}", json=json, headers=headers)


def poll(
    relative_url: str,
    error_field: Optional[str] = None,
    expected_code: Optional[int] = 200,
    headers: Optional[Headers] = None,
    url: str = URL,
) -> Response:
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


def poll_splits(dataset: str, config: Optional[str], headers: Optional[Headers] = None) -> Response:
    config_query = f"&config={config}" if config else ""
    return poll(f"/splits?dataset={dataset}{config_query}", error_field="error", headers=headers)


def poll_first_rows(dataset: str, config: str, split: str, headers: Optional[Headers] = None) -> Response:
    return poll(f"/first-rows?dataset={dataset}&config={config}&split={split}", error_field="error", headers=headers)


def get_openapi_body_example(path: str, status: int, example_name: str) -> Any:
    root = Path(__file__).resolve().parent.parent.parent
    openapi_filename = root / "docs" / "source" / "openapi.json"
    with open(openapi_filename) as json_file:
        openapi = json.load(json_file)
    steps = [
        "paths",
        path,
        "get",
        "responses",
        str(status),
        "content",
        "application/json",
        "examples",
        example_name,
        "value",
    ]
    result = openapi
    for step in steps:
        if "$ref" in result:
            new_steps = result["$ref"].split("/")[1:]
            result = openapi
            for new_step in new_steps:
                result = result[new_step]
        result = result[step]
    return result


def get_default_config_split() -> tuple[str, str]:
    config = "default"
    split = "train"
    return config, split


def log(response: Response, url: str = URL, relative_url: Optional[str] = None, dataset: Optional[str] = None) -> str:
    extra = ""
    if dataset is not None:
        try:
            extra_response = get(
                f"/admin/dataset-status?dataset={dataset}",
                headers={"Authorization": f"Bearer {E2E_ADMIN_USER_TOKEN}"},
                url=url,
            )
            if extra_response.status_code == 200:
                extra = f"content of dataset-status: {extra_response.text}"
            else:
                extra = f"cannot get content of dataset-status: {extra_response.status_code} - {extra_response.text}"
        except Exception as e:
            extra = f"cannot get content of dataset-status - {e}"
        extra = f"\n{extra}"
    return (
        f"{dataset=} - {relative_url=} - {response.status_code} - {response.headers} - {response.text} - {url}{extra}"
    )


def poll_until_ready_and_assert(
    relative_url: str,
    expected_status_code: int = 200,
    expected_error_code: Optional[str] = None,
    headers: Optional[Headers] = None,
    url: str = URL,
    check_x_revision: bool = False,
    dataset: str = None,
) -> Any:
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
    assert response.status_code == expected_status_code, log(response, url, relative_url, dataset)
    assert response.headers.get("X-Error-Code") == expected_error_code, log(response, url, relative_url)
    if check_x_revision:
        assert response.headers.get("X-Revision") is not None, log(response, url, relative_url)
        assert len(str(response.headers.get("X-Revision"))) == 40, log(response, url, relative_url)
    return response


def has_metric(name: str, labels: Mapping[str, str], metric_names: set[str]) -> bool:
    label_str = ",".join([f'{k}="{v}"' for k, v in sorted(labels.items())])
    s = name + "{" + label_str + "}"
    return any(re.match(s, metric_name) is not None for metric_name in metric_names)


def update_repo_settings(
    hf_api: HfApi,
    repo_id: str,
    *,
    private: Optional[bool] = None,
    gated: Optional[str] = None,
    token: Optional[str] = None,
    organization: Optional[str] = None,
    repo_type: Optional[str] = None,
    name: Optional[str] = None,
) -> Any:
    """Update the settings of a repository.
    Args:
        repo_id (`str`, *optional*):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
            <Tip>
            Version added: 0.5
            </Tip>
        private (`bool`, *optional*, defaults to `None`):
            Whether the repo should be private.
        gated (`str`, *optional*, defaults to `None`):
            Whether the repo should request user access.
            Possible values are 'auto' and 'manual'
        token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
        repo_type (`str`, *optional*):
            Set to `"dataset"` or `"space"` if uploading to a dataset or
            space, `None` or `"model"` if uploading to a model. Default is
            `None`.
    Returns:
        The HTTP response in json.
    <Tip>
    Raises the following errors:
        - [`~huggingface_hub.utils.RepositoryNotFoundError`]
            If the repository to download from cannot be found. This may be because it doesn't exist,
            or because it is set to `private` and you do not have access.
    </Tip>
    """
    if repo_type not in REPO_TYPES:
        raise ValueError("Invalid repo type")

    organization, name = repo_id.split("/") if "/" in repo_id else (None, repo_id)

    if organization is None:
        namespace = hf_api.whoami(token=token)["name"]
    else:
        namespace = organization

    path_prefix = f"{hf_api.endpoint}/api/"
    if repo_type in REPO_TYPES_URL_PREFIXES:
        path_prefix += REPO_TYPES_URL_PREFIXES[repo_type]

    path = f"{path_prefix}{namespace}/{name}/settings"

    json: dict[str, Union[bool, str]] = {}
    if private is not None:
        json["private"] = private
    if gated is not None:
        json["gated"] = gated

    r = requests.put(
        path,
        headers={"authorization": f"Bearer {token}"},
        json=json,
    )
    hf_raise_for_status(r)
    return r.json()


@contextmanager
def tmp_dataset(namespace: str, token: str, csv_path: str, private: bool, gated: Optional[str]) -> Iterator[str]:
    # create a test dataset in hub-ci, then delete it
    hf_api = HfApi(endpoint=CI_HUB_ENDPOINT, token=token)
    dataset = f"{namespace}/tmp-dataset-{int(time.time() * 10e3)}"
    hf_api.create_repo(
        repo_id=dataset,
        private=private,
        repo_type="dataset",
    )
    hf_api.upload_file(
        path_or_fileobj=csv_path,
        path_in_repo="data/csv_data.csv",
        repo_id=dataset,
        repo_type="dataset",
    )
    if gated:
        update_repo_settings(hf_api, dataset, gated=gated, repo_type="dataset")
    try:
        yield dataset
    finally:
        with suppress(requests.exceptions.HTTPError, ValueError):
            hf_api.delete_repo(repo_id=dataset, repo_type="dataset")


# explicit re-export
__all__ = ["Response"]
