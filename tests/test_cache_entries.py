import pytest

from datasets_preview_backend.cache import cache_directory  # type: ignore
from datasets_preview_backend.cache_entries import get_cache_entry
from datasets_preview_backend.exceptions import Status404Error
from datasets_preview_backend.queries.configs import get_configs


def test_cache_directory() -> None:
    # ensure the cache directory is empty, so that this file gets an empty cache
    assert cache_directory is None
    # note that the same cache is used all over this file. We might want to call
    # http://www.grantjenks.com/docs/diskcache/api.html#diskcache.Cache.clear
    # at the beginning of every test to start with an empty cache


def test_get_cache_entry_error() -> None:
    endpoint = "/configs"
    kwargs = {"dataset": "doesnotexist"}

    report = get_cache_entry(endpoint, get_configs, kwargs)
    assert report["status"] == "cache_miss"

    # warm the cache
    with pytest.raises(Status404Error):
        get_configs(**kwargs)

    report = get_cache_entry(endpoint, get_configs, kwargs)
    assert report["status"] == "error"
