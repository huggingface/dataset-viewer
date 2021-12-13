import pytest
from mongoengine import DoesNotExist

from datasets_preview_backend.config import MONGO_CACHE_DATABASE
from datasets_preview_backend.io.cache import (
    DbDataset,
    clean_database,
    connect_to_cache,
    delete_dataset_cache,
    refresh_dataset,
    upsert_dataset,
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
    dataset_cache = DbDataset(dataset_name="test", status="cache_miss")
    dataset_cache.save()

    retrieved = DbDataset.objects(dataset_name="test")
    assert len(list(retrieved)) == 1


def test_acronym_identification() -> None:
    dataset_name = "acronym_identification"
    dataset = get_dataset(dataset_name)
    upsert_dataset(dataset)
    retrieved = DbDataset.objects(dataset_name=dataset_name).get()
    assert retrieved.dataset_name == dataset_name
    assert retrieved.status == "valid"
    delete_dataset_cache(dataset_name)
    with pytest.raises(DoesNotExist):
        DbDataset.objects(dataset_name=dataset_name).get()


def test_doesnotexist() -> None:
    dataset_name = "doesnotexist"
    refresh_dataset(dataset_name)
    retrieved = DbDataset.objects(dataset_name=dataset_name).get()
    assert retrieved.status == "error"


def test_config_error() -> None:
    # see https://github.com/huggingface/datasets-preview-backend/issues/78
    dataset_name = "Check/region_1"
    refresh_dataset(dataset_name)
    retrieved = DbDataset.objects(dataset_name=dataset_name).get()
    assert retrieved.status == "error"


def test_large_document() -> None:
    # see https://github.com/huggingface/datasets-preview-backend/issues/89
    dataset_name = "SaulLu/Natural_Questions_HTML"
    refresh_dataset(dataset_name)
    retrieved = DbDataset.objects(dataset_name=dataset_name).get()
    assert retrieved.status == "valid"
