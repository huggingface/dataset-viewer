from .utils import poll


def test_healthcheck():
    # this tests ensures the /healthcheck and the /metrics endpoints are hidden
    response = poll("/healthcheck", expected_code=404)
    assert response.status_code == 404, f"{response.status_code} - {response.text}"
    assert "Not Found" in response.text, response.text

    response = poll("/metrics", expected_code=404)
    assert response.status_code == 404, f"{response.status_code} - {response.text}"
    assert "Not Found" in response.text, response.text
