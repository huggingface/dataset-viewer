import pytest

from .utils import (
    get_openapi_body_example,
    refresh_poll_splits_next,
)


@pytest.mark.parametrize(
    "dataset,status,name",
    [
        ("duorc", 200, "duorc"),
        ("emotion", 200, "emotion"),
        ("severo/inexistent-dataset", 404, "inexistent-dataset"),
        ("severo/dummy_private", 404, "private-dataset"),
    ],
)
def test_splits_next(dataset, status, name):
    body = get_openapi_body_example("/splits-next", status, name)

    r_splits = refresh_poll_splits_next(dataset)

    assert r_splits.status_code == status
    assert r_splits.json() == body
