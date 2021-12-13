import pytest

from datasets_preview_backend.config import MONGO_QUEUE_DATABASE
from datasets_preview_backend.io.queue import (
    EmptyQueue,
    InvalidJobId,
    JobNotFound,
    add_job,
    clean_database,
    connect_to_queue,
    finish_job,
    get_job,
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
    add_job("test")
    add_job("test")
    job_id, dataset_name = get_job()
    assert dataset_name == "test"
    add_job("test")
    with pytest.raises(EmptyQueue):
        get_job()
    finish_job(job_id)
    with pytest.raises(EmptyQueue):
        get_job()
    add_job("test")
    job_id, dataset_name = get_job()
    with pytest.raises(InvalidJobId):
        finish_job("invalid_id")
    with pytest.raises(JobNotFound):
        other_job_id = ("1" if job_id[0] == "0" else "0") + job_id[1:]
        finish_job(other_job_id)
    finish_job(job_id)
