from datasets_preview_backend.queries.cache_stats import get_cache_stats
from datasets_preview_backend.queries.configs import get_configs_response
from datasets_preview_backend.queries.datasets import get_datasets_response
from datasets_preview_backend.queries.info import get_info_response
from datasets_preview_backend.queries.rows import get_rows_response
from datasets_preview_backend.queries.splits import get_splits_response


def test_get_cache_stats() -> None:
    response = get_cache_stats()
    assert "endpoints" in response
    endpoints = response["endpoints"]
    assert len(endpoints) == 5
    endpoint = endpoints["/datasets"]
    assert endpoint["endpoint"] == "/datasets"
    assert endpoint["expected"] == 1
    assert endpoint["cached"] == 0
    assert endpoint["expired"] == 0
    assert endpoint["error"] == 0
    assert endpoint["valid"] == 0

    # add datasets to the cache
    get_datasets_response()
    response = get_cache_stats()
    endpoints = response["endpoints"]
    endpoint = endpoints["/datasets"]
    assert endpoint["endpoint"] == "/datasets"
    assert endpoint["expected"] == 1
    assert endpoint["cached"] == 1
    assert endpoint["expired"] == 0
    assert endpoint["error"] == 0
    assert endpoint["valid"] == 1
    endpoint = endpoints["/info"]
    assert endpoint["endpoint"] == "/info"
    assert endpoint["expected"] > 100
    assert endpoint["cached"] == 0
    assert endpoint["expired"] == 0
    assert endpoint["error"] == 0
    assert endpoint["valid"] == 0
    endpoint = endpoints["/configs"]
    assert endpoint["endpoint"] == "/configs"
    assert endpoint["expected"] > 100
    assert endpoint["cached"] == 0
    assert endpoint["expired"] == 0
    assert endpoint["error"] == 0
    assert endpoint["valid"] == 0

    # add info to the cache
    get_info_response(dataset="glue")
    response = get_cache_stats()
    endpoints = response["endpoints"]
    endpoint = endpoints["/info"]
    assert endpoint["endpoint"] == "/info"
    assert endpoint["cached"] == 1
    assert endpoint["expired"] == 0
    assert endpoint["error"] == 0
    assert endpoint["valid"] == 1

    # add info to the cache
    # this dataset is not in the list of datasets (see /datasets) and is thus not expected:
    # it does not appear in the stats, even if the response is in the cache
    info_response = get_info_response(dataset="doesnotexist")
    assert info_response.content is not None
    assert "exception" in info_response.content
    response = get_cache_stats()
    endpoints = response["endpoints"]
    endpoint = endpoints["/info"]
    assert endpoint["endpoint"] == "/info"
    assert endpoint["cached"] == 1
    assert endpoint["expired"] == 0
    assert endpoint["error"] == 0
    assert endpoint["valid"] == 1

    # add info to the cache
    # this dataset has no info, but is expected: it is in the cache and is erroneous
    info_response = get_info_response(dataset="jdepoix/junit_test_completion")
    assert info_response.content is not None
    assert "exception" in info_response.content
    response = get_cache_stats()
    endpoints = response["endpoints"]
    endpoint = endpoints["/info"]
    assert endpoint["endpoint"] == "/info"
    assert endpoint["cached"] == 2
    assert endpoint["expired"] == 0
    assert endpoint["error"] == 1
    assert endpoint["valid"] == 1

    # add configs to the cache
    get_configs_response(dataset="glue")
    response = get_cache_stats()
    endpoints = response["endpoints"]
    endpoint = endpoints["/configs"]
    assert endpoint["endpoint"] == "/configs"
    assert endpoint["cached"] == 1
    assert endpoint["expired"] == 0
    assert endpoint["error"] == 0
    assert endpoint["valid"] == 1
    endpoint = endpoints["/splits"]
    assert endpoint["endpoint"] == "/splits"
    assert endpoint["expected"] == 12
    assert endpoint["cached"] == 0
    assert endpoint["expired"] == 0
    assert endpoint["error"] == 0
    assert endpoint["valid"] == 0

    # add splits to the cache
    get_splits_response(dataset="glue", config="cola")
    response = get_cache_stats()
    endpoints = response["endpoints"]
    endpoint = endpoints["/splits"]
    assert endpoint["endpoint"] == "/splits"
    assert endpoint["cached"] == 1
    assert endpoint["expired"] == 0
    assert endpoint["error"] == 0
    assert endpoint["valid"] == 1
    endpoint = endpoints["/rows"]
    assert endpoint["endpoint"] == "/rows"
    assert endpoint["expected"] == 3
    assert endpoint["cached"] == 0
    assert endpoint["expired"] == 0
    assert endpoint["error"] == 0
    assert endpoint["valid"] == 0

    # add rows to the cache
    get_rows_response(dataset="glue", config="cola", split="train", num_rows=100)
    response = get_cache_stats()
    endpoints = response["endpoints"]
    endpoint = endpoints["/rows"]
    assert endpoint["endpoint"] == "/rows"
    assert endpoint["cached"] == 1
    assert endpoint["expired"] == 0
    assert endpoint["error"] == 0
    assert endpoint["valid"] == 1
