import pytest
from libcache.simple_cache import _clean_database as clean_cache_database
from libcache.simple_cache import connect_to_cache
from libqueue.queue import add_first_rows_job, add_splits_job
from libqueue.queue import clean_database as clean_queue_database
from libqueue.queue import connect_to_queue

from worker.main import process_next_first_rows_job, process_next_splits_job

from ._utils import MONGO_CACHE_DATABASE, MONGO_QUEUE_DATABASE, MONGO_URL


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


@pytest.mark.real_dataset
def test_process_next_splits_job():
    add_splits_job("acronym_identification")
    result = process_next_splits_job()
    assert result is True


@pytest.mark.real_dataset
def test_process_next_first_rows_job():
    add_first_rows_job("acronym_identification", "default", "train")
    result = process_next_first_rows_job()
    assert result is True
