from datasets_preview_backend.queries.cache_stats import get_cache_stats
from datasets_preview_backend.queries.configs import get_configs_response
from datasets_preview_backend.queries.datasets import get_datasets_response
from datasets_preview_backend.queries.infos import get_infos_response
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
    endpoint = endpoints["/configs"]
    assert endpoint["endpoint"] == "/configs"
    assert endpoint["expected"] > 100
    assert endpoint["cached"] == 0
    assert endpoint["expired"] == 0
    assert endpoint["error"] == 0
    assert endpoint["valid"] == 0

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

    # add infos to the cache
    get_infos_response(dataset="glue", config="ax")
    response = get_cache_stats()
    endpoints = response["endpoints"]
    endpoint = endpoints["/infos"]
    assert endpoint["endpoint"] == "/infos"
    assert endpoint["cached"] == 1
    assert endpoint["expired"] == 0
    assert endpoint["error"] == 0
    assert endpoint["valid"] == 1

    # add infos to the cache
    # this dataset is not in the list of datasets (see /datasets) and is thus not expected:
    # it does not appear in the stats, even if the response is in the cache
    infos_response = get_infos_response(dataset="doesnotexist", config="doesnotexist")
    assert infos_response.content is not None
    assert "exception" in infos_response.content
    response = get_cache_stats()
    endpoints = response["endpoints"]
    endpoint = endpoints["/infos"]
    assert endpoint["endpoint"] == "/infos"
    assert endpoint["cached"] == 1
    assert endpoint["expired"] == 0
    assert endpoint["error"] == 0
    assert endpoint["valid"] == 1

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
    get_rows_response(dataset="glue", config="cola", split="train")
    response = get_cache_stats()
    endpoints = response["endpoints"]
    endpoint = endpoints["/rows"]
    assert endpoint["endpoint"] == "/rows"
    assert endpoint["cached"] == 1
    assert endpoint["expired"] == 0
    assert endpoint["error"] == 0
    assert endpoint["valid"] == 1
