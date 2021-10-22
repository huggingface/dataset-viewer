from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.io.cache import cache_directory  # type: ignore
from datasets_preview_backend.models.info import get_info
from datasets_preview_backend.models.split import get_split, get_split_names


def test_cache_directory() -> None:
    # ensure the cache directory is empty, so that this file gets an empty cache
    assert cache_directory is None
    # note that the same cache is used all over this file. We might want to call
    # http://www.grantjenks.com/docs/diskcache/api.html#diskcache.Cache.clear
    # at the beginning of every test to start with an empty cache


# get_split
def test_get_split() -> None:
    info = get_info("glue", "ax")
    split = get_split("glue", "ax", "test", info)
    assert split["split_name"] == "test"
    assert "rows" in split
    assert "columns" in split


def test_get_splits_names() -> None:
    dataset = "acronym_identification"
    splits = get_split_names(dataset, DEFAULT_CONFIG_NAME)
    assert len(splits) == 3
    assert "train" in splits


def test_splits_fallback() -> None:
    # uses the fallback to call "builder._split_generators" while https://github.com/huggingface/datasets/issues/2743
    splits = get_split_names("hda_nli_hindi", "HDA nli hindi")
    assert len(splits) == 3
    assert "train" in splits
