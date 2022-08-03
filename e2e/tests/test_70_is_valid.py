import requests

from .utils import URL


def test_is_valid_after_datasets_processed():
    # this test ensures that a dataset processed successfully returns true in /is-valid
    response = requests.get(f"{URL}/is-valid")
    assert response.status_code == 422
    # at this moment various datasets have been processed (due to the alphabetic order of the test files)
    response = requests.get(f"{URL}/is-valid?dataset=acronym_identification")
    assert response.status_code == 200
    assert response.json()["valid"] is True
    # without authentication, we get a 401 error when requesting a non-existing dataset
    response = requests.get(f"{URL}/is-valid?dataset=non-existing-dataset")
    assert response.status_code == 401
