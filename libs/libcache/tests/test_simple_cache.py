import pytest
from pymongo.errors import DocumentTooLarge

from libcache.simple_cache import (
    DoesNotExist,
    HTTPStatus,
    _clean_database,
    connect_to_cache,
    delete_first_rows_responses,
    delete_splits_responses,
    get_first_rows_response,
    get_first_rows_response_reports,
    get_first_rows_responses_count_by_status,
    get_splits_response,
    get_splits_response_reports,
    get_splits_responses_count_by_status,
    get_valid_dataset_names,
    mark_first_rows_responses_as_stale,
    mark_splits_responses_as_stale,
    upsert_first_rows_response,
    upsert_splits_response,
)

from ._utils import MONGO_CACHE_DATABASE, MONGO_URL


@pytest.fixture(autouse=True, scope="module")
def safe_guard() -> None:
    if "test" not in MONGO_CACHE_DATABASE:
        raise ValueError("Test must be launched on a test mongo database")


@pytest.fixture(autouse=True, scope="module")
def client() -> None:
    connect_to_cache(database=MONGO_CACHE_DATABASE, host=MONGO_URL)


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    _clean_database()


def test_upsert_splits_response() -> None:
    dataset_name = "test_dataset"
    response = {"splits": [{"dataset_name": dataset_name, "config_name": "test_config", "split_name": "test_split"}]}
    upsert_splits_response(dataset_name, response, HTTPStatus.OK)
    response1, http_status = get_splits_response(dataset_name)
    assert http_status == HTTPStatus.OK
    assert response1 == response

    # ensure it's idempotent
    upsert_splits_response(dataset_name, response, HTTPStatus.OK)
    (response2, _) = get_splits_response(dataset_name)
    assert response2 == response1

    mark_splits_responses_as_stale(dataset_name)
    # we don't have access to the stale field
    # we also don't have access to the updated_at field

    delete_splits_responses(dataset_name)
    with pytest.raises(DoesNotExist):
        get_splits_response(dataset_name)

    mark_splits_responses_as_stale(dataset_name)
    with pytest.raises(DoesNotExist):
        get_splits_response(dataset_name)


def test_upsert_first_rows_response() -> None:
    dataset_name = "test_dataset"
    config_name = "test_config"
    split_name = "test_split"
    response = {"key": "value"}
    upsert_first_rows_response(dataset_name, config_name, split_name, response, HTTPStatus.OK)
    response1, http_status = get_first_rows_response(dataset_name, config_name, split_name)
    assert http_status == HTTPStatus.OK
    assert response1 == response

    # ensure it's idempotent
    upsert_first_rows_response(dataset_name, config_name, split_name, response, HTTPStatus.OK)
    (response2, _) = get_first_rows_response(dataset_name, config_name, split_name)
    assert response2 == response1

    mark_first_rows_responses_as_stale(dataset_name)
    mark_first_rows_responses_as_stale(dataset_name, config_name, split_name)
    # we don't have access to the stale field
    # we also don't have access to the updated_at field

    upsert_first_rows_response(dataset_name, config_name, "test_split2", response, HTTPStatus.OK)
    delete_first_rows_responses(dataset_name, config_name, "test_split2")
    get_first_rows_response(dataset_name, config_name, split_name)

    delete_first_rows_responses(dataset_name)
    with pytest.raises(DoesNotExist):
        get_first_rows_response(dataset_name, config_name, split_name)

    mark_first_rows_responses_as_stale(dataset_name)
    mark_first_rows_responses_as_stale(dataset_name, config_name, split_name)
    with pytest.raises(DoesNotExist):
        get_first_rows_response(dataset_name, config_name, split_name)


def test_big_row() -> None:
    # https://github.com/huggingface/datasets-server/issues/197
    dataset_name = "test_dataset"
    config_name = "test_config"
    split_name = "test_split"
    big_response = {"content": "a" * 100_000_000}
    with pytest.raises(DocumentTooLarge):
        upsert_first_rows_response(dataset_name, config_name, split_name, big_response, HTTPStatus.OK)


def test_valid() -> None:
    assert get_valid_dataset_names() == []

    upsert_splits_response(
        "test_dataset",
        {"key": "value"},
        HTTPStatus.OK,
    )

    assert get_valid_dataset_names() == []

    upsert_first_rows_response(
        "test_dataset",
        "test_config",
        "test_split",
        {
            "key": "value",
        },
        HTTPStatus.OK,
    )

    assert get_valid_dataset_names() == ["test_dataset"]

    upsert_splits_response(
        "test_dataset2",
        {"key": "value"},
        HTTPStatus.OK,
    )

    assert get_valid_dataset_names() == ["test_dataset"]

    upsert_first_rows_response(
        "test_dataset2",
        "test_config2",
        "test_split2",
        {
            "key": "value",
        },
        HTTPStatus.BAD_REQUEST,
    )

    assert get_valid_dataset_names() == ["test_dataset"]

    upsert_first_rows_response(
        "test_dataset2",
        "test_config2",
        "test_split3",
        {
            "key": "value",
        },
        HTTPStatus.OK,
    )

    assert get_valid_dataset_names() == ["test_dataset", "test_dataset2"]


def test_count_by_status() -> None:
    assert get_splits_responses_count_by_status() == {"OK": 0, "BAD_REQUEST": 0, "INTERNAL_SERVER_ERROR": 0}

    upsert_splits_response(
        "test_dataset2",
        {"key": "value"},
        HTTPStatus.OK,
    )

    assert get_splits_responses_count_by_status() == {"OK": 1, "BAD_REQUEST": 0, "INTERNAL_SERVER_ERROR": 0}
    assert get_first_rows_responses_count_by_status() == {"OK": 0, "BAD_REQUEST": 0, "INTERNAL_SERVER_ERROR": 0}

    upsert_first_rows_response(
        "test_dataset",
        "test_config",
        "test_split",
        {
            "key": "value",
        },
        HTTPStatus.OK,
    )

    assert get_first_rows_responses_count_by_status() == {"OK": 1, "BAD_REQUEST": 0, "INTERNAL_SERVER_ERROR": 0}


def test_reports() -> None:
    assert get_splits_response_reports() == []
    upsert_splits_response(
        "a",
        {"key": "value"},
        HTTPStatus.OK,
    )
    upsert_splits_response(
        "b",
        {
            "error": "Cannot get the split names for the dataset.",
            "cause_exception": "FileNotFoundError",
            "cause_message": (
                "Couldn't find a dataset script at /src/services/worker/wikimedia/timit_asr/timit_asr.py or any data"
                " file in the same directory. Couldn't find 'wikimedia/timit_asr' on the Hugging Face Hub either:"
                " FileNotFoundError: Dataset 'wikimedia/timit_asr' doesn't exist on the Hub. If the repo is private,"
                " make sure you are authenticated with `use_auth_token=True` after logging in with `huggingface-cli"
                " login`."
            ),
            "cause_traceback": [
                "Traceback (most recent call last):\n",
                '  File "/src/services/worker/src/worker/models/dataset.py", line 17, in'
                " get_dataset_split_full_names\n    for config_name in get_dataset_config_names(dataset_name,"
                " use_auth_token=hf_token)\n",
                '  File "/src/services/worker/.venv/lib/python3.9/site-packages/datasets/inspect.py", line 289, in'
                " get_dataset_config_names\n    dataset_module = dataset_module_factory(\n",
                '  File "/src/services/worker/.venv/lib/python3.9/site-packages/datasets/load.py", line 1242, in'
                " dataset_module_factory\n    raise FileNotFoundError(\n",
                "FileNotFoundError: Couldn't find a dataset script at"
                " /src/services/worker/wikimedia/timit_asr/timit_asr.py or any data file in the same directory."
                " Couldn't find 'wikimedia/timit_asr' on the Hugging Face Hub either: FileNotFoundError: Dataset"
                " 'wikimedia/timit_asr' doesn't exist on the Hub. If the repo is private, make sure you are"
                " authenticated with `use_auth_token=True` after logging in with `huggingface-cli login`.\n",
            ],
        },
        HTTPStatus.BAD_REQUEST,
    )
    upsert_splits_response(
        "c",
        {
            "error": "cannot write mode RGBA as JPEG",
        },
        HTTPStatus.INTERNAL_SERVER_ERROR,
        {
            "status_code": 500,
            "message": "cannot write mode RGBA as JPEG",
            "cause_exception": "FileNotFoundError",
            "cause_message": (
                "Couldn't find a dataset script at /src/services/worker/wikimedia/timit_asr/timit_asr.py or any data"
                " file in the same directory. Couldn't find 'wikimedia/timit_asr' on the Hugging Face Hub either:"
                " FileNotFoundError: Dataset 'wikimedia/timit_asr' doesn't exist on the Hub. If the repo is private,"
                " make sure you are authenticated with `use_auth_token=True` after logging in with `huggingface-cli"
                " login`."
            ),
            "cause_traceback": [
                "Traceback (most recent call last):\n",
                '  File "/src/services/worker/src/worker/models/dataset.py", line 17, in'
                " get_dataset_split_full_names\n    for config_name in get_dataset_config_names(dataset_name,"
                " use_auth_token=hf_token)\n",
                '  File "/src/services/worker/.venv/lib/python3.9/site-packages/datasets/inspect.py", line 289, in'
                " get_dataset_config_names\n    dataset_module = dataset_module_factory(\n",
                '  File "/src/services/worker/.venv/lib/python3.9/site-packages/datasets/load.py", line 1242, in'
                " dataset_module_factory\n    raise FileNotFoundError(\n",
                "FileNotFoundError: Couldn't find a dataset script at"
                " /src/services/worker/wikimedia/timit_asr/timit_asr.py or any data file in the same directory."
                " Couldn't find 'wikimedia/timit_asr' on the Hugging Face Hub either: FileNotFoundError: Dataset"
                " 'wikimedia/timit_asr' doesn't exist on the Hub. If the repo is private, make sure you are"
                " authenticated with `use_auth_token=True` after logging in with `huggingface-cli login`.\n",
            ],
        },
    )
    assert get_splits_response_reports() == [
        {"dataset": "a", "error": None, "status": "200"},
        {
            "dataset": "b",
            "error": {
                "cause_exception": "FileNotFoundError",
                "message": "Cannot get the split names for the dataset.",
            },
            "status": "400",
        },
        {"dataset": "c", "error": {"message": "cannot write mode RGBA as JPEG"}, "status": "500"},
    ]

    assert get_first_rows_response_reports() == []
