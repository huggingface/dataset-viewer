from .utils import poll


def test_healthcheck():
    # this tests ensures the nginx reverse proxy and the api are up
    response = poll("/healthcheck", expected_code=404)
    assert response.status_code == 404, f"{response.status_code} - {response.text}"
    assert "Not Found" in response.text, response.text
