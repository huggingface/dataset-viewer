from datasets_preview_backend.cache_reports import get_kwargs_report
from datasets_preview_backend.config import CACHE_DIRECTORY, CACHE_PERSIST
from datasets_preview_backend.responses import get_cached_response


def test_cache_directory() -> None:
    # ensure the cache directory is empty, so that this file gets an empty cache
    assert CACHE_DIRECTORY is None
    assert not CACHE_PERSIST
    # note that the same cache is used all over this file. We might want to call
    # http://www.grantjenks.com/docs/diskcache/api.html#diskcache.Cache.clear
    # at the beginning of every test to start with an empty cache


def test_get_kwargs_report_error() -> None:
    endpoint = "/configs"
    kwargs = {"dataset": "doesnotexist"}

    report = get_kwargs_report(endpoint, kwargs)
    assert report["status"] == "cache_miss"

    # warm the cache
    get_cached_response(endpoint, **kwargs)
    report = get_kwargs_report(endpoint, kwargs)
    assert report["status"] == "error"

    report = get_kwargs_report(endpoint, kwargs)
    assert report["status"] == "error"
