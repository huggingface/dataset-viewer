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
    "status,name,dataset",
    [
        (200, "duorc", "duorc"),
        (200, "emotion", "emotion"),
        (404, "inexistent-dataset", "severo/inexistent-dataset"),
        (404, "private-dataset", "severo/dummy_private"),
        (422, "empty-parameter", ""),
        (422, "missing-parameter", None),
        (500, "SplitsNotFoundError", "natural_questions"),
        (500, "FileNotFoundError", "akhaliq/test"),
        (500, "not-ready", "a_new_dataset"),
        # not tested: 'internal_error'
    ],
)
def test_splits_next(status, name, dataset):
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
