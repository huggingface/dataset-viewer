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

from ._utils import (
    ASSETS_BASE_URL,
    MONGO_CACHE_DATABASE,
    MONGO_QUEUE_DATABASE,
    MONGO_URL,
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
    assert refresh_splits(dataset_name) == (HTTPStatus.NOT_FOUND, False)
    with pytest.raises(DoesNotExist):
        get_splits_response(dataset_name)


def test_e2e_examples() -> None:
    # see https://github.com/huggingface/datasets-server/issues/78
    dataset_name = "Check/region_1"

    assert refresh_splits(dataset_name) == (HTTPStatus.OK, False)
    response, _, _ = get_splits_response(dataset_name)
    assert len(response["splits"]) == 1
    assert response["splits"][0]["num_bytes"] is None
    assert response["splits"][0]["num_examples"] is None

    dataset_name = "acronym_identification"
    assert refresh_splits(dataset_name) == (HTTPStatus.OK, False)
    response, _, _ = get_splits_response(dataset_name)
    assert len(response["splits"]) == 3
    assert response["splits"][0]["num_bytes"] == 7792803
    assert response["splits"][0]["num_examples"] == 14006


def test_large_document() -> None:
    # see https://github.com/huggingface/datasets-server/issues/89
    dataset_name = "SaulLu/Natural_Questions_HTML"

    assert refresh_splits(dataset_name) == (HTTPStatus.OK, False)
    _, http_status, error_code = get_splits_response(dataset_name)
    assert http_status == HTTPStatus.OK
    assert error_code is None


def test_first_rows() -> None:
    http_status, _ = refresh_first_rows("common_voice", "tr", "train", ASSETS_BASE_URL)
    response, cached_http_status, error_code = get_first_rows_response("common_voice", "tr", "train")
    assert http_status == HTTPStatus.OK
    assert cached_http_status == HTTPStatus.OK
    assert error_code is None

    assert response["features"][0]["feature_idx"] == 0
    assert response["features"][0]["name"] == "client_id"
    assert response["features"][0]["type"]["_type"] == "Value"
    assert response["features"][0]["type"]["dtype"] == "string"

    assert response["features"][2]["name"] == "audio"
    assert response["features"][2]["type"]["_type"] == "Audio"
    assert response["features"][2]["type"]["sampling_rate"] == 48000

    assert response["rows"][0]["row_idx"] == 0
    assert response["rows"][0]["row"]["client_id"].startswith("54fc2d015c27a057b")
    assert response["rows"][0]["row"]["audio"] == [
        {"src": f"{ASSETS_BASE_URL}/common_voice/--/tr/train/0/audio/audio.mp3", "type": "audio/mpeg"},
        {"src": f"{ASSETS_BASE_URL}/common_voice/--/tr/train/0/audio/audio.wav", "type": "audio/wav"},
    ]
