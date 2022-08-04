from .utils import poll


def test_healthcheck():
    # this tests ensures the nginx reverse proxy and the api are up
    response = poll("/healthcheck")
    assert response.status_code == 200, f"{response.status_code} - {response.text}"
    assert response.text == "ok", response.text
