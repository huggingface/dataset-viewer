# import pytest
import requests
import time

URL = "http://localhost:8080"


def get_response(url: str, timeout: int = 20) -> requests.Response:
    interval = 1
    retries = timeout // interval
    done = False
    while retries > 0 and not done:
        retries -= 1
        time.sleep(interval)
        response = requests.get(url)
        # print(response.text)
        done = response.headers.get("Content-Type") == "application/json"
    return response


def get_splits_response(dataset: str, timeout: int = 20) -> requests.Response:
    return get_response(f"{URL}/splits?dataset={dataset}", timeout)


def get_rows_response(dataset: str, config: str, split: str, timeout: int = 50) -> requests.Response:
    return get_response(f"{URL}/rows?dataset={dataset}&config={config}&split={split}", timeout)


def test_healthcheck():
    response = requests.get(f"{URL}/healthcheck")
    assert response.status_code == 200


def test_get_dataset():
    dataset = "acronym_identification"
    config = "default"
    split = "train"
    response = requests.post(f"{URL}/webhook", json={"update": f"datasets/{dataset}"})
    assert response.status_code == 200

    response = get_splits_response(dataset, timeout=15)
    assert response.status_code == 200

    response = get_rows_response(dataset, config, split, timeout=15)
    assert response.status_code == 200
    json = response.json()
    assert "rows" in json
    assert json["rows"][0]["row"]["id"] == "TR-0"
