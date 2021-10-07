from datasets_preview_backend.cache import cache_directory  # type: ignore
from datasets_preview_backend.cache_entries import get_cache_entry, memoized_functions
from datasets_preview_backend.responses import get_cached_response


def test_cache_directory() -> None:
    # ensure the cache directory is empty, so that this file gets an empty cache
    assert cache_directory is None
    # note that the same cache is used all over this file. We might want to call
    # http://www.grantjenks.com/docs/diskcache/api.html#diskcache.Cache.clear
    # at the beginning of every test to start with an empty cache


def test_get_cache_entry_error() -> None:
    endpoint = "/configs"
    kwargs = {"dataset": "doesnotexist"}

    report = get_cache_entry(endpoint, kwargs)
    assert report["status"] == "cache_miss"

    # warm the cache
    get_cached_response(memoized_functions, endpoint, **kwargs)

    report = get_cache_entry(endpoint, kwargs)
    assert report["status"] == "error"
