import pytest

from libqueue.queue import (
    EmptyQueue,
    add_first_rows_job,
    add_splits_job,
    clean_database,
    connect_to_queue,
    finish_splits_job,
    get_first_rows_job,
    get_first_rows_jobs_count_by_status,
    get_splits_job,
    get_splits_jobs_count_by_status,
)

from ._utils import MONGO_QUEUE_DATABASE, MONGO_URL


@pytest.fixture(autouse=True, scope="module")
def safe_guard() -> None:
    if "test" not in MONGO_QUEUE_DATABASE:
        raise ValueError("Test must be launched on a test mongo database")


@pytest.fixture(autouse=True, scope="module")
def client() -> None:
    connect_to_queue(database=MONGO_QUEUE_DATABASE, host=MONGO_URL)


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    clean_database()


def test_add_job() -> None:
    # add a job
    add_splits_job("test")
    # a second call is ignored
    add_splits_job("test")
    # get and start the first job
    job_id, dataset_name, retries = get_splits_job()
    assert dataset_name == "test"
    assert retries == 0
    # adding the job while the first one has not finished yet is ignored
    add_splits_job("test")
    with pytest.raises(EmptyQueue):
        # thus: no new job available
        get_splits_job()
    # finish the first job
    finish_splits_job(job_id, success=True)
    # the queue is empty
    with pytest.raises(EmptyQueue):
        get_splits_job()
    # add a job again
    add_splits_job("test", retries=5)
    # get it and start it
    job_id, dataset_name, retries = get_splits_job()
    other_job_id = ("1" if job_id[0] == "0" else "0") + job_id[1:]
    assert retries == 5
    finish_splits_job(other_job_id, success=True)
    # ^ fails silently (with a log)
    finish_splits_job(job_id, success=True)


def test_max_jobs_per_dataset() -> None:
    add_first_rows_job("dataset", "config", "split1")
    add_first_rows_job("dataset", "config", "split2")
    add_first_rows_job("dataset", "config", "split3")
    _, dataset_name, config_name, split_name, __ = get_first_rows_job()
    assert dataset_name == "dataset"
    assert config_name == "config"
    assert split_name == "split1"
    with pytest.raises(EmptyQueue):
        get_first_rows_job(0)
    with pytest.raises(EmptyQueue):
        get_first_rows_job(1)
    _, dataset_name, config_name, split_name, __ = get_first_rows_job(2)
    assert split_name == "split2"
    with pytest.raises(EmptyQueue):
        get_first_rows_job(2)


def test_count_by_status() -> None:
    assert get_splits_jobs_count_by_status() == {
        "waiting": 0,
        "started": 0,
        "success": 0,
        "error": 0,
        "cancelled": 0,
    }

    add_splits_job("test_dataset")

    assert get_splits_jobs_count_by_status() == {"waiting": 1, "started": 0, "success": 0, "error": 0, "cancelled": 0}

    assert get_first_rows_jobs_count_by_status() == {
        "waiting": 0,
        "started": 0,
        "success": 0,
        "error": 0,
        "cancelled": 0,
    }

    add_first_rows_job("test_dataset", "test_config", "test_split")

    assert get_first_rows_jobs_count_by_status() == {
        "waiting": 1,
        "started": 0,
        "success": 0,
        "error": 0,
        "cancelled": 0,
    }
