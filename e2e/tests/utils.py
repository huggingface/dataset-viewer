import json
import os
import time
from os.path import dirname, join
from typing import Optional, Tuple

import requests

SERVICE_REVERSE_PROXY_PORT = os.environ.get("SERVICE_REVERSE_PROXY_PORT", "8000")
SERVICE_ADMIN_PORT = os.environ.get("SERVICE_ADMIN_PORT", "8001")
SERVICE_API_PORT = os.environ.get("SERVICE_API_PORT", "8002")
ROWS_MAX_NUMBER = int(os.environ.get("ROWS_MAX_NUMBER", 100))
INTERVAL = 1
MAX_DURATION = 10 * 60
URL_REVERSE_PROXY = f"http://localhost:{SERVICE_REVERSE_PROXY_PORT}"
URL_ADMIN = f"http://localhost:{SERVICE_ADMIN_PORT}"
URL_API = f"http://localhost:{SERVICE_API_PORT}"
URL = URL_REVERSE_PROXY


def poll(url: str, error_field: Optional[str] = None, expected_code: Optional[int] = 200) -> requests.Response:
    interval = INTERVAL
    timeout = MAX_DURATION
    retries = timeout // interval
    should_retry = True
    response = None
    while retries > 0 and should_retry:
        retries -= 1
        time.sleep(interval)
        response = requests.get(url)
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


def post_refresh(dataset: str) -> requests.Response:
    return requests.post(f"{URL}/webhook", json={"update": f"datasets/{dataset}"})


def poll_splits(dataset: str) -> requests.Response:
    return poll(f"{URL}/splits?dataset={dataset}", error_field="message")


def poll_rows(dataset: str, config: str, split: str) -> requests.Response:
    return poll(f"{URL}/rows?dataset={dataset}&config={config}&split={split}", error_field="message")


def refresh_poll_splits_rows(dataset: str, config: str, split: str) -> Tuple[requests.Response, requests.Response]:
    # ask for the dataset to be refreshed
    response = post_refresh(dataset)
    assert response.status_code == 200

    # poll the /splits endpoint until we get something else than "The dataset is being processed. Retry later."
    response_splits = poll_splits(dataset)
    assert response.status_code == 200

    # poll the /rows endpoint until we get something else than "The split is being processed. Retry later."
    response_rows = poll_rows(dataset, config, split)
    assert response.status_code == 200

    return response_splits, response_rows


def poll_splits_next(dataset: str) -> requests.Response:
    return poll(f"{URL}/splits-next?dataset={dataset}", error_field="error")


def poll_first_rows(dataset: str, config: str, split: str) -> requests.Response:
    return poll(f"{URL}/first-rows?dataset={dataset}&config={config}&split={split}", error_field="error")


def refresh_poll_splits_next(dataset: str) -> requests.Response:
    # ask for the dataset to be refreshed
    response = post_refresh(dataset)
    assert response.status_code == 200

    # poll the /splits endpoint until we get something else than "The dataset is being processed. Retry later."
    return poll_splits_next(dataset)


def refresh_poll_splits_next_first_rows(
    dataset: str, config: str, split: str
) -> Tuple[requests.Response, requests.Response]:
    response_splits = refresh_poll_splits_next(dataset)
    assert response_splits.status_code == 200

    response_rows = poll_first_rows(dataset, config, split)

    return response_splits, response_rows


def get_openapi_body_example(path, status, example_name):
    root = dirname(dirname(dirname(__file__)))
    openapi_filename = join(root, "chart", "static-files", "openapi.json")
    with open(openapi_filename) as json_file:
        openapi = json.load(json_file)
    return openapi["paths"][path]["get"]["responses"][str(status)]["content"]["application/json"]["examples"][
        example_name
    ]["value"]
