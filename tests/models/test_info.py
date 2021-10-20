from datasets_preview_backend.io.cache import cache_directory  # type: ignore
from datasets_preview_backend.models.info import get_info


def test_cache_directory() -> None:
    # ensure the cache directory is empty, so that this file gets an empty cache
    assert cache_directory is None
    # note that the same cache is used all over this file. We might want to call
    # http://www.grantjenks.com/docs/diskcache/api.html#diskcache.Cache.clear
    # at the beginning of every test to start with an empty cache


def test_get_info() -> None:
    info = get_info("glue", "ax")
    assert "features" in info


def test_get_info_no_dataset_info_file() -> None:
    info = get_info("lhoestq/custom_squad", "plain_text")
    assert "features" in info
