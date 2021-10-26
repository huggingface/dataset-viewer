import pytest
from mongoengine import DoesNotExist

from datasets_preview_backend.config import MONGO_CACHE_DATABASE
from datasets_preview_backend.io.cache import (
    DatasetCache,
    clean_database,
    connect_cache,
    delete_dataset_cache,
    update_dataset_cache,
)


@pytest.fixture(autouse=True, scope="module")
def safe_guard() -> None:
    if "test" not in MONGO_CACHE_DATABASE:
        raise Exception("Test must be launched on a test mongo database")


@pytest.fixture(autouse=True, scope="module")
def client() -> None:
    connect_cache()


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    clean_database()


def test_save() -> None:
    dataset_cache = DatasetCache(dataset_name="test", status="cache_miss", content={"key": "value"})
    dataset_cache.save()

    retrieved = DatasetCache.objects(dataset_name="test")
    assert len(list(retrieved)) >= 1


def test_acronym_identification() -> None:
    dataset_name = "acronym_identification"
    update_dataset_cache(dataset_name)
    retrieved = DatasetCache.objects(dataset_name=dataset_name).get()
    assert retrieved.dataset_name == dataset_name
    assert len(retrieved.content["configs"]) == 1
    delete_dataset_cache(dataset_name)
    with pytest.raises(DoesNotExist):
        DatasetCache.objects(dataset_name=dataset_name).get()


def test_doesnotexist() -> None:
    dataset_name = "doesnotexist"
    update_dataset_cache(dataset_name)
    retrieved = DatasetCache.objects(dataset_name=dataset_name).get()
    assert retrieved.status == "error"
