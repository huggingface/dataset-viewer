from .utils import get


def test_valid_after_datasets_processed():
    # this test ensures that the datasets processed successfully are present in /valid
    response = get("/valid")
    assert response.status_code == 200, f"{response.status_code} - {response.text}"
    # at this moment various datasets have been processed (due to the alphabetic order of the test files)
    assert "acronym_identification" in response.json()["valid"], response.text
    assert "nielsr/CelebA-faces" in response.json()["valid"], response.text
