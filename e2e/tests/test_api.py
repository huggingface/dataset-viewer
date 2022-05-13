# import pytest
import requests

URL = "http://localhost:8000"


def test_healthcheck():
    response = requests.get(f"{URL}/healthcheck")
    assert response.status_code == 200


def test_get_dataset():
    response = requests.post(f"{URL}/webhook", json={"update": "datasets/acronym_identification"})
    assert response.status_code == 200

    response = requests.get(f"{URL}/prometheus")
    lines = response.text.split("\n")
    metrics = {line.split(" ")[0]: float(line.split(" ")[1]) for line in lines if line and line[0] != "#"}
    assert (
        metrics['queue_jobs_total{queue="datasets",status="waiting"}']
        + metrics['queue_jobs_total{queue="datasets",status="started"}']
        + metrics['queue_jobs_total{queue="datasets",status="success"}']
        >= 1
    )
