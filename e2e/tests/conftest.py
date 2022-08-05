import pytest

from .utils import poll


# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.files", "tests.fixtures.hub"]


@pytest.fixture(autouse=True, scope="session")
def ensure_services_are_up() -> None:
    assert poll("/", expected_code=404).status_code == 404
    assert poll("/healthcheck").status_code == 200
    assert poll("/admin/healthcheck").status_code == 200
    # TODO: add endpoints to check the workers are up?
