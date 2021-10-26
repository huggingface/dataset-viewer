import pytest
from datetime import datetime

from datasets_preview_backend.config import MONGO_QUEUE_DATABASE
from datasets_preview_backend.io.queue import (  # DatasetCache,; delete_dataset_cache,; update_dataset_cache,
    clean_database,
    connect_queue,
    Job,
)

# from mongoengine import DoesNotExist


@pytest.fixture(autouse=True, scope="module")
def safe_guard() -> None:
    if "test" not in MONGO_QUEUE_DATABASE:
        raise Exception("Test must be launched on a test mongo database")


@pytest.fixture(autouse=True, scope="module")
def client() -> None:
    connect_queue()


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    clean_database()


def test_save() -> None:
    job = Job(dataset_name="test", priority=1, start_time=datetime.utcnow(), end_time=None)
    job.save()

    retrieved = Job.objects(dataset_name="test")
    assert len(list(retrieved)) == 1
