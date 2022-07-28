import requests

from .utils import URL


def test_valid_after_datasets_processed():
    # this test ensures that the datasets processed successfully are present in /valid
    response = requests.get(f"{URL}/valid")
    assert response.status_code == 200
    # at this moment various datasets have been processed
    assert "acronym_identification" in response.json()["valid"]
    assert "nielsr/CelebA-faces" in response.json()["valid"]
