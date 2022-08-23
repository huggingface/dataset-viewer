from http import HTTPStatus

import pytest
from libcache.simple_cache import DoesNotExist
from libcache.simple_cache import _clean_database as clean_cache_database
from libcache.simple_cache import (
    connect_to_cache,
    get_first_rows_response,
    get_splits_response,
)
from libqueue.queue import clean_database as clean_queue_database
from libqueue.queue import connect_to_queue

from worker.refresh import refresh_first_rows, refresh_splits

from .fixtures.files import DATA
from .utils import (
    ASSETS_BASE_URL,
    DEFAULT_HF_ENDPOINT,
    HF_ENDPOINT,
    MONGO_CACHE_DATABASE,
    MONGO_QUEUE_DATABASE,
    MONGO_URL,
    ROWS_MAX_NUMBER,
    get_default_config_split,
)


@pytest.fixture(autouse=True, scope="module")
def safe_guard() -> None:
    if "test" not in MONGO_CACHE_DATABASE:
        raise ValueError("Test must be launched on a test mongo database")


@pytest.fixture(autouse=True, scope="module")
def client() -> None:
    connect_to_cache(database=MONGO_CACHE_DATABASE, host=MONGO_URL)
    connect_to_queue(database=MONGO_QUEUE_DATABASE, host=MONGO_URL)


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    clean_cache_database()
    clean_queue_database()


def test_doesnotexist() -> None:
    dataset_name = "doesnotexist"
    assert refresh_splits(dataset_name, hf_endpoint=HF_ENDPOINT) == (HTTPStatus.NOT_FOUND, False)
    with pytest.raises(DoesNotExist):
        get_splits_response(dataset_name)
    dataset, config, split = get_default_config_split(dataset_name)
    assert refresh_first_rows(dataset, config, split, ASSETS_BASE_URL, hf_endpoint=HF_ENDPOINT) == (
        HTTPStatus.NOT_FOUND,
        False,
    )
    with pytest.raises(DoesNotExist):
        get_first_rows_response(dataset, config, split)


def test_refresh_splits(hf_public_dataset_repo_csv_data: str) -> None:
    assert refresh_splits(hf_public_dataset_repo_csv_data, hf_endpoint=HF_ENDPOINT) == (HTTPStatus.OK, False)
    response, _, _ = get_splits_response(hf_public_dataset_repo_csv_data)
    assert len(response["splits"]) == 1
    assert response["splits"][0]["num_bytes"] is None
    assert response["splits"][0]["num_examples"] is None


def test_refresh_first_rows(hf_public_dataset_repo_csv_data: str) -> None:
    dataset, config, split = get_default_config_split(hf_public_dataset_repo_csv_data)
    http_status, _ = refresh_first_rows(dataset, config, split, ASSETS_BASE_URL, hf_endpoint=HF_ENDPOINT)
    response, cached_http_status, error_code = get_first_rows_response(dataset, config, split)
    assert http_status == HTTPStatus.OK
    assert cached_http_status == HTTPStatus.OK
    assert error_code is None
    assert response["features"][0]["feature_idx"] == 0
    assert response["features"][0]["name"] == "col_1"
    assert response["features"][0]["type"]["_type"] == "Value"
    assert response["features"][0]["type"]["dtype"] == "int64"  # <---|
    assert response["features"][1]["type"]["dtype"] == "int64"  # <---|- auto-detected by the datasets library
    assert response["features"][2]["type"]["dtype"] == "float64"  # <-|

    assert len(response["rows"]) == min(len(DATA), ROWS_MAX_NUMBER)
    assert response["rows"][0]["row_idx"] == 0
    assert response["rows"][0]["row"] == {"col_1": 0, "col_2": 0, "col_3": 0.0}


@pytest.mark.real_dataset
def test_large_document() -> None:
    # see https://github.com/huggingface/datasets-server/issues/89
    dataset_name = "SaulLu/Natural_Questions_HTML"

    assert refresh_splits(dataset_name, hf_endpoint=DEFAULT_HF_ENDPOINT) == (HTTPStatus.OK, False)
    _, http_status, error_code = get_splits_response(dataset_name)
    assert http_status == HTTPStatus.OK
    assert error_code is None
