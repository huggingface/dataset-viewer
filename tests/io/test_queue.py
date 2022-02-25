import pytest

from datasets_preview_backend.config import MONGO_QUEUE_DATABASE
from datasets_preview_backend.io.queue import (
    EmptyQueue,
    JobNotFound,
    add_dataset_job,
    clean_database,
    connect_to_queue,
    finish_dataset_job,
    get_dataset_job,
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
