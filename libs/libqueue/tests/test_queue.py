import pytest

from libqueue.queue import (
    EmptyQueue,
    FirstRowsJob,
    Status,
    add_first_rows_job,
    add_splits_job,
    clean_database,
    connect_to_queue,
    finish_first_rows_job,
    finish_splits_job,
    get_datetime,
    get_first_rows_job,
    get_first_rows_jobs_count_by_status,
    get_splits_job,
    get_splits_jobs_count_by_status,
    is_first_rows_response_in_process,
    is_splits_response_in_process,
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
    assert is_splits_response_in_process("test") is True
    # get and start the first job
    job_id, dataset_name, retries = get_splits_job()
    assert dataset_name == "test"
    assert retries == 0
    assert is_splits_response_in_process("test") is True
    # adding the job while the first one has not finished yet is ignored
    add_splits_job("test")
    with pytest.raises(EmptyQueue):
        # thus: no new job available
        get_splits_job()
    # finish the first job
    finish_splits_job(job_id, success=True)
    # the queue is empty
    assert is_splits_response_in_process("test") is False
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


def test_add_job_with_broken_collection() -> None:
    dataset_name = "dataset_broken"
    config_name = "config_broken"
    split_name = "split_broken"
    # ensure the jobs are cancelled with more than one exist in a "pending" status
    # we "manually" create two jobs in a "pending" status for the same split
    # (we normally cannot do that with the exposed methods)
    job_1 = FirstRowsJob(
        dataset_name=dataset_name,
        config_name=config_name,
        split_name=split_name,
        created_at=get_datetime(),
        status=Status.WAITING,
    ).save()
    job_2 = FirstRowsJob(
        dataset_name=dataset_name,
        config_name=config_name,
        split_name=split_name,
        created_at=get_datetime(),
        started_at=get_datetime(),
        status=Status.STARTED,
    ).save()
    # then we add a job: it should create a new job in the "WAITING" status
    # and the two other jobs should be cancelled
    add_first_rows_job(dataset_name=dataset_name, config_name=config_name, split_name=split_name)
    assert (
        FirstRowsJob.objects(
            dataset_name=dataset_name, config_name=config_name, split_name=split_name, status__in=[Status.WAITING]
        ).count()
        == 1
    )
    assert FirstRowsJob.objects(pk=job_1.pk).get().status == Status.CANCELLED
    assert FirstRowsJob.objects(pk=job_2.pk).get().status == Status.CANCELLED


def test_max_jobs_per_dataset() -> None:
    add_first_rows_job("dataset", "config", "split1")
    assert is_first_rows_response_in_process("dataset", "config", "split1") is True
    add_first_rows_job("dataset", "config", "split2")
    add_first_rows_job("dataset", "config", "split3")
    job_id, dataset_name, config_name, split_name, __ = get_first_rows_job()
    assert dataset_name == "dataset"
    assert config_name == "config"
    assert split_name == "split1"
    assert is_first_rows_response_in_process("dataset", "config", "split1") is True
    with pytest.raises(EmptyQueue):
        get_first_rows_job(0)
    with pytest.raises(EmptyQueue):
        get_first_rows_job(1)
    _, dataset_name, config_name, split_name, __ = get_first_rows_job(2)
    assert split_name == "split2"
    with pytest.raises(EmptyQueue):
        get_first_rows_job(2)
    # finish the first job
    finish_first_rows_job(job_id, success=True)
    assert is_first_rows_response_in_process("dataset", "config", "split1") is False


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
