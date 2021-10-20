from datasets_preview_backend.io.cache import cache_directory  # type: ignore
from datasets_preview_backend.models.hf_dataset import get_hf_datasets


def test_cache_directory() -> None:
    # ensure the cache directory is empty, so that this file gets an empty cache
    assert cache_directory is None
    # note that the same cache is used all over this file. We might want to call
    # http://www.grantjenks.com/docs/diskcache/api.html#diskcache.Cache.clear
    # at the beginning of every test to start with an empty cache


# get_dataset_names
def test_get_dataset_items() -> None:
    dataset_items = get_hf_datasets()
    assert len(dataset_items) > 1000
    glue_datasets = [dataset for dataset in dataset_items if dataset["id"] == "glue"]
    assert len(glue_datasets) == 1
    glue = glue_datasets[0]
    assert glue["id"] == "glue"
    assert len(glue["tags"]) > 5
    assert "task_categories:text-classification" in glue["tags"]
    assert glue["citation"] is not None
    assert glue["citation"].startswith("@inproceedings")
    assert glue["description"] is not None
    assert glue["description"].startswith("GLUE, the General Language")
    assert glue["paperswithcode_id"] is not None
    assert glue["paperswithcode_id"] == "glue"
    assert glue["downloads"] is not None
    assert glue["downloads"] > 500000
