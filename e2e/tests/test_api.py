# import pytest
import time

import requests

URL = "http://localhost:8080"


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
    interval = 1
    timeout = 15
    url = f"{URL}/splits?dataset={dataset}"
    retries = timeout // interval
    done = False
    response = None
    json = None
    while retries > 0 and not done:
        retries -= 1
        time.sleep(interval)
        response = requests.get(url)
        json = response.json()
        done = not json or "message" not in json or json["message"] != "The dataset is being processed. Retry later."
    assert response.status_code == 200

    interval = 1
    timeout = 15
    url = f"{URL}/rows?dataset={dataset}&config={config}&split={split}"
    retries = timeout // interval
    done = False
    response = None
    json = None
    while retries > 0 and not done:
        retries -= 1
        time.sleep(interval)
        response = requests.get(url)
        json = response.json()
        done = not json or "message" not in json or json["message"] != "The split is being processed. Retry later."
    assert response.status_code == 200
    assert "rows" in json
    assert json["rows"][0]["row"]["id"] == "TR-0"
