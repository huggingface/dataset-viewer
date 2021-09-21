from time import time
from typing import Any, List, cast, Dict, Union, Optional

from datasets_preview_backend.queries.configs import get_configs_response
from datasets_preview_backend.queries.datasets import get_datasets_response
from datasets_preview_backend.queries.info import get_info_response
from datasets_preview_backend.queries.rows import get_rows_response
from datasets_preview_backend.queries.splits import get_splits_response
from datasets_preview_backend.types import (
    ArgsCacheStats,
    CacheStats,
    ConfigsDict,
    DatasetsDict,
    EndpointCacheStats,
    SplitsDict,
)


def get_endpoint_report(endpoint: str, args_reports: List[ArgsCacheStats]) -> EndpointCacheStats:
    expected = args_reports
    cached = [d for d in expected if d["is_cached"]]
    valid = [d for d in cached if not d["is_expired"]]
    return {"endpoint": endpoint, "expected": len(expected), "cached": len(cached), "valid": len(valid)}


def get_kwargs_report(memoized_function: Any, kwargs: Any) -> ArgsCacheStats:
    cache = memoized_function.__cache__
    # cache.close()
    key = memoized_function.__cache_key__(**kwargs)
    response, expire_time = cache.get(key, default=None, expire_time=True)
    is_cached = response is not None
    is_expired = is_cached and expire_time is not None and expire_time < time()
    return {
        "kwargs": kwargs,
        "is_cached": is_cached,
        "is_expired": is_expired,
        "content": None if response is None else response.content,
    }


def get_cache_stats() -> CacheStats:
    endpoints = {}

    datasets_kwargs_list: Any = [{}]
    datasets_reports = [get_kwargs_report(get_datasets_response, kwargs) for kwargs in datasets_kwargs_list]
    endpoints["/datasets"] = get_endpoint_report("/datasets", datasets_reports)

    valid_datasets_reports = [d for d in datasets_reports if d["is_cached"] and not d["is_expired"]]
    for datasets_report in valid_datasets_reports:
        datasets_dict = cast(DatasetsDict, datasets_report["content"])
        datasets = datasets_dict["datasets"]

        info_kwargs_list = [{"dataset": dataset} for dataset in datasets]
        info_reports = [get_kwargs_report(get_info_response, kwargs) for kwargs in info_kwargs_list]
        endpoints["/info"] = get_endpoint_report("/info", info_reports)

        configs_kwargs_list = info_kwargs_list
        configs_reports = [get_kwargs_report(get_configs_response, kwargs) for kwargs in configs_kwargs_list]
        endpoints["/configs"] = get_endpoint_report("/configs", configs_reports)

        valid_configs_reports = [d for d in configs_reports if d["is_cached"] and not d["is_expired"]]
        for configs_report in valid_configs_reports:
            configs_dict = cast(ConfigsDict, configs_report["content"])
            dataset = configs_dict["dataset"]
            configs = configs_dict["configs"]

            splits_kwargs_list = [{"dataset": dataset, "config": config} for config in configs]
            splits_reports = [get_kwargs_report(get_splits_response, kwargs) for kwargs in splits_kwargs_list]
            endpoints["/splits"] = get_endpoint_report("/splits", splits_reports)

            valid_splits_reports = [d for d in splits_reports if d["is_cached"] and not d["is_expired"]]
            for splits_report in valid_splits_reports:
                splits_dict = cast(SplitsDict, splits_report["content"])
                dataset = splits_dict["dataset"]
                config = splits_dict["config"]
                splits = splits_dict["splits"]

                # TODO: manage the num_rows argument
                rows_args_list = [
                    {"dataset": dataset, "config": config, "split": split, "num_rows": 100} for split in splits
                ]
                rows_reports = [get_kwargs_report(get_rows_response, args) for args in rows_args_list]
                endpoints["/rows"] = get_endpoint_report("/rows", rows_reports)

    return {"endpoints": endpoints}
