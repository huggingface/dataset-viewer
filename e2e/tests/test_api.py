# import pytest
import requests

URL = "http://localhost:8000"


def test_healthcheck():
    response = requests.get(f"{URL}/healthcheck")
    assert response.status_code == 200


def test_get_dataset():
    response = requests.post(f"{URL}/webhook", json={"update": "datasets/acronym_identification"})
    assert response.status_code == 200

    response = requests.get(f"{URL}/queue")
    json = response.json()
    datasets = json["datasets"]
    assert datasets["waiting"] + datasets["started"] + datasets["success"] >= 1
