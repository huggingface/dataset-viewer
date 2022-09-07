import pytest

from .fixtures.hub import AuthHeaders, AuthType, DatasetRepos, DatasetReposType
from .utils import (
    Response,
    get,
    get_default_config_split,
    poll_first_rows,
    refresh_poll_splits,
)


def log(response: Response, dataset: str) -> str:
    dataset, config, split = get_default_config_split(dataset)
    return f"{response.status_code} - {response.text} - {dataset} - {config} - {split}"


@pytest.mark.parametrize(
    "type,auth,status_code,error_code_splits,error_code_first_rows",
    [
        ("public", "none", 200, None, None),
        ("public", "token", 200, None, None),
        ("public", "cookie", 200, None, None),
        ("gated", "none", 401, "ExternalUnauthenticatedError", "ExternalUnauthenticatedError"),
        ("gated", "token", 200, None, None),
        ("gated", "cookie", 200, None, None),
        ("private", "none", 401, "ExternalUnauthenticatedError", "ExternalUnauthenticatedError"),
        ("private", "token", 404, "SplitsResponseNotFound", "FirstRowsResponseNotFound"),
        ("private", "cookie", 404, "SplitsResponseNotFound", "FirstRowsResponseNotFound"),
    ],
)
def test_split_public_auth(
    auth_headers: AuthHeaders,
    hf_dataset_repos_csv_data: DatasetRepos,
    type: DatasetReposType,
    auth: AuthType,
    status_code: int,
    error_code_splits: str,
    error_code_first_rows: str,
) -> None:
    if auth not in auth_headers:
        # ignore the test case if the auth type is not configured
        pytest.skip(f"auth {auth} has not been configured")
    dataset, config, split = get_default_config_split(hf_dataset_repos_csv_data[type])
    # pivate: no need to refresh, it's not implemented.
    # TODO: the webhook should respond 501 Not implemented when provided with a private dataset
    # (and delete the cache if existing)
    r_splits = (
        get(f"/splits?dataset={dataset}", headers=auth_headers[auth])
        if type == "private"
        else refresh_poll_splits(dataset, headers=auth_headers[auth])
    )
    assert r_splits.status_code == status_code, log(r_splits, dataset)
    assert r_splits.headers.get("X-Error-Code") == error_code_splits, log(r_splits, dataset)

    r_rows = (
        get(f"/first-rows?dataset={dataset}&config={config}&split={split}", headers=auth_headers[auth])
        if type == "private"
        else poll_first_rows(dataset, config, split, headers=auth_headers[auth])
    )
    assert r_rows.status_code == status_code, log(r_rows, dataset)
    assert r_rows.headers.get("X-Error-Code") == error_code_first_rows, log(r_rows, dataset)
