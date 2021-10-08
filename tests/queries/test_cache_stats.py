import pytest

from datasets_preview_backend.exceptions import Status404Error
from datasets_preview_backend.queries.cache_stats import get_cache_stats
from datasets_preview_backend.queries.configs import get_configs
from datasets_preview_backend.queries.datasets import get_datasets
from datasets_preview_backend.queries.infos import get_infos
from datasets_preview_backend.queries.rows import get_rows
from datasets_preview_backend.queries.splits import get_splits


def test_get_cache_stats() -> None:
    response = get_cache_stats()
    assert "endpoints" in response
    endpoints = response["endpoints"]
    assert len(endpoints) == 5
    endpoint = endpoints["/datasets"]
    assert endpoint["endpoint"] == "/datasets"
    assert endpoint["expected"] == 1
    assert endpoint["valid"] == 0
    assert endpoint["error"] == 0
    assert endpoint["cache_expired"] == 0
    assert endpoint["cache_miss"] == 1

    # add datasets to the cache
    get_datasets()
    response = get_cache_stats()
    endpoints = response["endpoints"]
    endpoint = endpoints["/datasets"]
    assert endpoint["endpoint"] == "/datasets"
    assert endpoint["expected"] == 1
    assert endpoint["cache_miss"] == 0
    assert endpoint["cache_expired"] == 0
    assert endpoint["error"] == 0
    assert endpoint["valid"] == 1
    endpoint = endpoints["/configs"]
    assert endpoint["endpoint"] == "/configs"
    assert endpoint["expected"] > 100
    assert endpoint["valid"] == 0
    assert endpoint["error"] == 0
    assert endpoint["cache_expired"] == 0
    assert endpoint["cache_miss"] > 100

    # add configs to the cache
    get_configs(dataset="glue")
    response = get_cache_stats()
    endpoints = response["endpoints"]
    endpoint = endpoints["/configs"]
    assert endpoint["endpoint"] == "/configs"
    assert endpoint["valid"] == 1
    assert endpoint["error"] == 0
    assert endpoint["cache_expired"] == 0
    assert endpoint["cache_miss"] > 100
    endpoint = endpoints["/splits"]
    assert endpoint["endpoint"] == "/splits"
    assert endpoint["expected"] == 12
    assert endpoint["valid"] == 0
    assert endpoint["error"] == 0
    assert endpoint["cache_expired"] == 0
    assert endpoint["cache_miss"] == 12

    # add infos to the cache
    get_infos(dataset="glue", config="ax")
    response = get_cache_stats()
    endpoints = response["endpoints"]
    endpoint = endpoints["/infos"]
    assert endpoint["endpoint"] == "/infos"
    assert endpoint["valid"] == 1
    assert endpoint["error"] == 0
    assert endpoint["cache_expired"] == 0
    assert endpoint["cache_miss"] == 11

    # add infos to the cache
    # this dataset is not in the list of datasets (see /datasets) and is thus not expected:
    # it does not appear in the stats, even if the response is in the cache
    with pytest.raises(Status404Error):
        get_infos(dataset="doesnotexist", config="doesnotexist")
    response = get_cache_stats()
    endpoints = response["endpoints"]
    endpoint = endpoints["/infos"]
    assert endpoint["endpoint"] == "/infos"
    assert endpoint["valid"] == 1
    assert endpoint["error"] == 0
    assert endpoint["cache_expired"] == 0
    assert endpoint["cache_miss"] == 11

    # add splits to the cache
    get_splits(dataset="glue", config="cola")
    response = get_cache_stats()
    endpoints = response["endpoints"]
    endpoint = endpoints["/splits"]
    assert endpoint["endpoint"] == "/splits"
    assert endpoint["valid"] == 1
    assert endpoint["error"] == 0
    assert endpoint["cache_expired"] == 0
    assert endpoint["cache_miss"] == 11
    endpoint = endpoints["/rows"]
    assert endpoint["endpoint"] == "/rows"
    assert endpoint["expected"] == 3
    assert endpoint["valid"] == 0
    assert endpoint["error"] == 0
    assert endpoint["cache_expired"] == 0
    assert endpoint["cache_miss"] == 3

    # add rows to the cache
    get_rows(dataset="glue", config="cola", split="train")
    response = get_cache_stats()
    endpoints = response["endpoints"]
    endpoint = endpoints["/rows"]
    assert endpoint["endpoint"] == "/rows"
    assert endpoint["valid"] == 1
    assert endpoint["error"] == 0
    assert endpoint["cache_expired"] == 0
    assert endpoint["cache_miss"] == 2
