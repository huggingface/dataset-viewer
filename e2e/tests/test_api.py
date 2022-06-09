# import pytest
import time

import requests

URL = "http://localhost:8080"
# ^ we should access localhost:8000 (reverse-proxy) instead of 8080 (api)
# but there seems to be a bug: the errors seem to be cached by the reverse proxy


def poll_splits_until_dataset_process_has_finished(
    dataset: str, timeout: int = 15, interval: int = 1
) -> requests.Response:
    url = f"{URL}/splits?dataset={dataset}"
    retries = timeout // interval
    done = False
    response = None
    while retries > 0 and not done:
        retries -= 1
        time.sleep(interval)
        response = requests.get(url)
        json = response.json()
        done = not json or "message" not in json or json["message"] != "The dataset is being processed. Retry later."
    if response is None:
        raise RuntimeError("no request has been done")
    return response


def poll_rows_until_split_process_has_finished(
    dataset: str, config: str, split: str, timeout: int = 15, interval: int = 1
) -> requests.Response:
    url = f"{URL}/rows?dataset={dataset}&config={config}&split={split}"
    retries = timeout // interval
    done = False
    response = None
    while retries > 0 and not done:
        retries -= 1
        time.sleep(interval)
        response = requests.get(url)
        json = response.json()
        done = not json or "message" not in json or json["message"] != "The split is being processed. Retry later."
    if response is None:
        raise RuntimeError("no request has been done")
    return response


def test_healthcheck():
    response = requests.get(f"{URL}/healthcheck")
    assert response.status_code == 200


def test_get_dataset():
    dataset = "acronym_identification"
    config = "default"
    split = "train"

    # ask for the dataset to be refreshed
    response = requests.post(f"{URL}/webhook", json={"update": f"datasets/{dataset}"})
    assert response.status_code == 200

    # poll the /splits endpoint until we get something else than "The dataset cache is empty."
    response = poll_splits_until_dataset_process_has_finished(dataset, 15)
    assert response.status_code == 200

    # poll the /rows endpoint until we get something else than "The split cache is empty."
    response = poll_rows_until_split_process_has_finished(dataset, config, split, 15)
    assert response.status_code == 200
    json = response.json()
    assert "rows" in json
    assert json["rows"][0]["row"]["id"] == "TR-0"


def test_reproduce_bug_empty_split():
    # see #185 and #177
    # we get an error when:
    # - the dataset has been processed and the splits have been created in the database
    # - the splits have not been processed and are still in EMPTY status in the database
    # - the dataset is processed again, and the splits are marked as STALLED
    # - as STALLED, they are thus returned with an empty content, instead of an error message (waiting for being processsed)
    dataset = "nielsr/CelebA-faces"
    config = "nielsr--CelebA-faces"
    split = "train"

    # ask for the dataset to be refreshed
    response = requests.post(f"{URL}/webhook", json={"update": f"datasets/{dataset}"})
    assert response.status_code == 200

    # poll the /splits endpoint until we get something else than "The dataset cache is empty."
    response = poll_splits_until_dataset_process_has_finished(dataset, 15)
    assert response.status_code == 200

    # at this point the splits should have been created in the dataset, and still be EMPTY
    url = f"{URL}/rows?dataset={dataset}&config={config}&split={split}"
    response = requests.get(url)
    assert response.status_code == 400
    json = response.json()
    assert json["message"] == "The split is being processed. Retry later."

    # ask again for the dataset to be refreshed
    response = requests.post(f"{URL}/webhook", json={"update": f"datasets/{dataset}"})
    assert response.status_code == 200

    # at this moment, there is a concurrency race between the dataset worker and the split worker
    # but the dataset worker should finish before, because it's faster on this dataset
    # if we poll again /rows until we have something else than "being processed", we should
    # get a valid response, but with empty rows, which is incorrect and due to the bug
    response = poll_rows_until_split_process_has_finished(dataset, config, split, 15)
    assert response.status_code == 200
    json = response.json()
    assert json == {"columns": [], "rows": []}
