import pytest

from .utils import URL, URL_ADMIN, poll


@pytest.fixture(autouse=True, scope="session")
def ensure_services_are_up() -> None:
    assert poll(f"{URL}/", expected_code=404).status_code == 404
    assert poll(f"{URL}/healthcheck").status_code == 200
    assert poll(f"{URL_ADMIN}/healthcheck").status_code == 200
    # TODO: add endpoints to check the workers are up?
