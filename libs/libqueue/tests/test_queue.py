import pytest

from libqueue.config import MONGO_QUEUE_DATABASE
from libqueue.queue import (
    EmptyQueue,
    add_dataset_job,
    add_split_job,
    clean_database,
    connect_to_queue,
    finish_dataset_job,
    get_dataset_job,
    is_dataset_in_queue,
    is_split_in_queue,
)


@pytest.fixture(autouse=True, scope="module")
def safe_guard() -> None:
    if "test" not in MONGO_QUEUE_DATABASE:
        raise Exception("Test must be launched on a test mongo database")


@pytest.fixture(autouse=True, scope="module")
def client() -> None:
    connect_to_queue()


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    clean_database()


def test_add_job() -> None:
    add_dataset_job("test")
    add_dataset_job("test")
    job_id, dataset_name = get_dataset_job()
    assert dataset_name == "test"
    add_dataset_job("test")
    with pytest.raises(EmptyQueue):
        get_dataset_job()
    finish_dataset_job(job_id, success=True)
    with pytest.raises(EmptyQueue):
        get_dataset_job()
    add_dataset_job("test")
    job_id, dataset_name = get_dataset_job()
    other_job_id = ("1" if job_id[0] == "0" else "0") + job_id[1:]
    finish_dataset_job(other_job_id, success=True)
    # ^ fails silently (with a log)
    finish_dataset_job(job_id, success=True)


def test_is_dataset_in_queue() -> None:
    dataset_name = "test_dataset"
    dataset_name_2 = "test_dataset_2"
    assert is_dataset_in_queue(dataset_name) is False
    add_dataset_job(dataset_name)
    assert is_dataset_in_queue(dataset_name) is True
    assert is_dataset_in_queue(dataset_name_2) is False


def test_is_split_in_queue() -> None:
    dataset_name = "test_dataset"
    config_name = "test_config"
    split_name = "test_split"
    dataset_name_2 = "test_dataset_2"
    assert is_split_in_queue(dataset_name, config_name, split_name) is False
    add_split_job(dataset_name, config_name, split_name)
    assert is_split_in_queue(dataset_name, config_name, split_name) is True
    assert is_split_in_queue(dataset_name_2, config_name, split_name) is False
