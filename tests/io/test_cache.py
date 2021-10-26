import pytest
from mongoengine import DoesNotExist

from datasets_preview_backend.config import MONGO_CACHE_DATABASE
from datasets_preview_backend.exceptions import StatusError
from datasets_preview_backend.io.cache import (
    DatasetCache,
    clean_database,
    connect_to_cache,
    delete_dataset_cache,
    upsert_dataset_cache,
)
from datasets_preview_backend.models.dataset import get_dataset


@pytest.fixture(autouse=True, scope="module")
def safe_guard() -> None:
    if "test" not in MONGO_CACHE_DATABASE:
        raise Exception("Test must be launched on a test mongo database")


@pytest.fixture(autouse=True, scope="module")
def client() -> None:
    connect_to_cache()


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    clean_database()


def test_save() -> None:
    dataset_cache = DatasetCache(dataset_name="test", status="cache_miss", content={"key": "value"})
    dataset_cache.save()

    retrieved = DatasetCache.objects(dataset_name="test")
    assert len(list(retrieved)) == 1


def test_acronym_identification() -> None:
    dataset_name = "acronym_identification"
    dataset = get_dataset(dataset_name)
    upsert_dataset_cache(dataset_name, "valid", dataset)
    retrieved = DatasetCache.objects(dataset_name=dataset_name).get()
    assert retrieved.dataset_name == dataset_name
    assert len(retrieved.content["configs"]) == 1
    delete_dataset_cache(dataset_name)
    with pytest.raises(DoesNotExist):
        DatasetCache.objects(dataset_name=dataset_name).get()


def test_doesnotexist() -> None:
    dataset_name = "doesnotexist"
    try:
        get_dataset(dataset_name)
    except StatusError as err:
        upsert_dataset_cache(dataset_name, "error", err.as_content())
    retrieved = DatasetCache.objects(dataset_name=dataset_name).get()
    assert retrieved.status == "error"
