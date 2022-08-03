import pytest
import requests

from .utils import (
    URL,
    get_openapi_body_example,
    poll,
    post_refresh,
    refresh_poll_splits_next,
)


@pytest.mark.parametrize(
    "status,name,dataset,error_code",
    [
        (200, "duorc", "duorc", None),
        (200, "emotion", "emotion", None),
        (
            401,
            "inexistent-dataset",
            "severo/inexistent-dataset",
            "ExternalUnauthenticatedError",
        ),
        (
            401,
            "gated-dataset",
            "severo/dummy_gated",
            "ExternalUnauthenticatedError",
        ),
        (
            401,
            "private-dataset",
            "severo/dummy_private",
            "ExternalUnauthenticatedError",
        ),
        (422, "empty-parameter", "", "MissingRequiredParameter"),
        (422, "missing-parameter", None, "MissingRequiredParameter"),
        (500, "SplitsNotFoundError", "natural_questions", "SplitsNamesError"),
        (500, "FileNotFoundError", "akhaliq/test", "SplitsNamesError"),
        (500, "not-ready", "severo/fix-401", "SplitsResponseNotReady"),
        # not tested: 'internal_error'
    ],
)
def test_splits_next(status: int, name: str, dataset: str, error_code: str):
    body = get_openapi_body_example("/splits-next", status, name)

    if name == "empty-parameter":
        r_splits = poll(f"{URL}/splits-next?dataset=", error_field="error")
    elif name == "missing-parameter":
        r_splits = poll(f"{URL}/splits-next", error_field="error")
    elif name == "not-ready":
        post_refresh(dataset)
        # poll the endpoint before the worker had the chance to process it
        r_splits = requests.get(f"{URL}/splits-next?dataset={dataset}")
    else:
        r_splits = refresh_poll_splits_next(dataset)

    assert r_splits.status_code == status
    assert r_splits.json() == body
    if error_code is not None:
        assert r_splits.headers["X-Error-Code"] == error_code
    else:
        assert "X-Error-Code" not in r_splits.headers
