from .utils import get


def test_is_valid_after_datasets_processed():
    # this test ensures that a dataset processed successfully returns true in /is-valid
    response = get("/is-valid")
    assert response.status_code == 422, f"{response.status_code} - {response.text}"
    # at this moment various datasets have been processed (due to the alphabetic order of the test files)
    response = get("/is-valid?dataset=acronym_identification")
    assert response.status_code == 200, f"{response.status_code} - {response.text}"
    assert response.json()["valid"] is True, response.text
    # without authentication, we get a 401 error when requesting a non-existing dataset
    response = get("/is-valid?dataset=non-existing-dataset")
    assert response.status_code == 401, f"{response.status_code} - {response.text}"
