import pytest

from .fixtures.hub import AuthHeaders, AuthType, DatasetRepos, DatasetReposType

from .utils import (
    get,
    get_openapi_body_example,
    poll,
    post_refresh,
    refresh_poll_splits_next,
)


@pytest.mark.parametrize(
    "type,auth,status_code,error_code",
    [
        ("public", "none", 200, None),
        ("public", "token", 200, None),
        ("public", "cookie", 200, None),
        ("gated", "none", 401, "ExternalUnauthenticatedError"),
        ("gated", "token", 200, None),
        ("gated", "cookie", 200, None),
        ("private", "none", 401, "ExternalUnauthenticatedError"),
        ("private", "token", 404, "SplitsResponseNotFound"),
        ("private", "cookie", 404, "SplitsResponseNotFound"),
    ],
)
def test_splits_next_public_auth(
    auth_headers: AuthHeaders,
    hf_dataset_repos_csv_data: DatasetRepos,
    type: DatasetReposType,
    auth: AuthType,
    status_code: int,
    error_code: str,
) -> None:
    if auth not in auth_headers:
        # ignore the test case if the auth type is not configured
        pytest.skip(f"auth {auth} has not been configured")
    if type == "private":
        # no need to refresh, it's not implemented.
        # TODO: the webhook should respond 501 Not implemented when provided with a private dataset
        # (and delete the cache if existing)
        response = get(f"/splits-next?dataset={hf_dataset_repos_csv_data[type]}", headers=auth_headers[auth])
    else:
        response = refresh_poll_splits_next(hf_dataset_repos_csv_data[type], headers=auth_headers[auth])
    assert (
        response.status_code == status_code
    ), f"{response.status_code} - {response.text} - {hf_dataset_repos_csv_data[type]}"
    assert (
        response.headers.get("X-Error-Code") == error_code
    ), f"{response.status_code} - {response.text} - {hf_dataset_repos_csv_data[type]}"


@pytest.mark.parametrize(
    "status,name,dataset,error_code",
    [
        # (200, "duorc", "duorc", None),
        # (200, "emotion", "emotion", None),
        (
            401,
            "inexistent-dataset",
            "severo/inexistent-dataset",
            "ExternalUnauthenticatedError",
        ),
        # (
        #     401,
        #     "gated-dataset",
        #     "severo/dummy_gated",
        #     "ExternalUnauthenticatedError",
        # ),
        # (
        #     401,
        #     "private-dataset",
        #     "severo/dummy_private",
        #     "ExternalUnauthenticatedError",
        # ),
        (422, "empty-parameter", "", "MissingRequiredParameter"),
        (422, "missing-parameter", None, "MissingRequiredParameter"),
        # (500, "SplitsNotFoundError", "natural_questions", "SplitsNamesError"),
        # (500, "FileNotFoundError", "akhaliq/test", "SplitsNamesError"),
        # (500, "not-ready", "severo/fix-401", "SplitsResponseNotReady"),
        # not tested: 'internal_error'
    ],
)
def test_splits_next(status: int, name: str, dataset: str, error_code: str):
    body = get_openapi_body_example("/splits-next", status, name)

    if name == "empty-parameter":
        r_splits = poll("/splits-next?dataset=", error_field="error")
    elif name == "missing-parameter":
        r_splits = poll("/splits-next", error_field="error")
    elif name == "not-ready":
        post_refresh(dataset)
        # poll the endpoint before the worker had the chance to process it
        r_splits = get(f"/splits-next?dataset={dataset}")
    else:
        r_splits = refresh_poll_splits_next(dataset)

    assert r_splits.status_code == status, f"{r_splits.status_code} - {r_splits.text}"
    assert r_splits.json() == body, r_splits.text
    if error_code is not None:
        assert r_splits.headers["X-Error-Code"] == error_code, r_splits.headers["X-Error-Code"]
    else:
        assert "X-Error-Code" not in r_splits.headers, r_splits.headers["X-Error-Code"]
