from time import time
from typing import Any, Dict, List, TypedDict, Union, cast

from datasets_preview_backend.queries.configs import get_configs_response
from datasets_preview_backend.queries.datasets import get_datasets_response
from datasets_preview_backend.queries.info import get_info_response
from datasets_preview_backend.queries.rows import get_rows_response
from datasets_preview_backend.queries.splits import get_splits_response
from datasets_preview_backend.types import (
    ConfigsDict,
    DatasetsDict,
    ResponseContent,
    SplitsDict,
)


class ArgsCacheStats(TypedDict):
    kwargs: Dict[str, Union[str, int]]
    is_cached: bool
    is_expired: bool
    is_error: bool
    is_valid: bool
    content: Union[ResponseContent, None]


class EndpointCacheStats(TypedDict):
    endpoint: str
    expected: int
    cached: int
    expired: int
    error: int
    valid: int


class CacheStats(TypedDict):
    endpoints: Dict[str, EndpointCacheStats]


def get_endpoint_report(endpoint: str, args_reports: List[ArgsCacheStats]) -> EndpointCacheStats:
    expected = args_reports
    cached = [d for d in expected if d["is_cached"]]
    expired = [d for d in expected if d["is_expired"]]
    error = [d for d in expected if d["is_error"]]
    valid = [d for d in expected if d["is_valid"]]
    return {
        "endpoint": endpoint,
        "expected": len(expected),
        "cached": len(cached),
        "expired": len(expired),
        "error": len(error),
        "valid": len(valid),
    }


def get_kwargs_report(memoized_function: Any, kwargs: Any) -> ArgsCacheStats:
    cache = memoized_function.__cache__
    # cache.close()
    key = memoized_function.__cache_key__(**kwargs)
    response, expire_time = cache.get(key, default=None, expire_time=True)
    is_cached = response is not None
    is_expired = expire_time is not None and expire_time < time()
    is_error = response is not None and response.is_error()
    is_valid = is_cached and not is_expired and not is_error
    return {
        "kwargs": kwargs,
        "is_cached": is_cached,
        "is_expired": is_expired,
        "is_error": is_error,
        "is_valid": is_valid,
        "content": None if response is None else response.content,
    }


def get_cache_stats() -> CacheStats:
    endpoints = {}

    datasets_kwargs_list: Any = [{}]
    datasets_reports = [get_kwargs_report(get_datasets_response, kwargs) for kwargs in datasets_kwargs_list]
    info_reports: List[ArgsCacheStats] = []
    configs_reports: List[ArgsCacheStats] = []
    splits_reports: List[ArgsCacheStats] = []
    rows_reports: List[ArgsCacheStats] = []

    valid_datasets_reports = [d for d in datasets_reports if d["is_valid"]]
    for datasets_report in valid_datasets_reports:
        datasets_dict = cast(DatasetsDict, datasets_report["content"])
        datasets = datasets_dict["datasets"]

        info_kwargs_list = [{"dataset": dataset["dataset"]} for dataset in datasets]
        local_info_reports = [get_kwargs_report(get_info_response, kwargs) for kwargs in info_kwargs_list]
        info_reports += local_info_reports

        configs_kwargs_list = info_kwargs_list
        local_configs_reports = [get_kwargs_report(get_configs_response, kwargs) for kwargs in configs_kwargs_list]
        configs_reports += local_configs_reports

        valid_configs_reports = [d for d in local_configs_reports if d["is_valid"]]
        for configs_report in valid_configs_reports:
            configs_dict = cast(ConfigsDict, configs_report["content"])
            dataset = configs_dict["dataset"]
            configs = configs_dict["configs"]

            splits_kwargs_list = [{"dataset": dataset, "config": config} for config in configs]
            local_splits_reports = [get_kwargs_report(get_splits_response, kwargs) for kwargs in splits_kwargs_list]
            splits_reports += local_splits_reports

            valid_splits_reports = [d for d in local_splits_reports if d["is_valid"]]
            for splits_report in valid_splits_reports:
                splits_dict = cast(SplitsDict, splits_report["content"])
                dataset = splits_dict["dataset"]
                config = splits_dict["config"]
                splits = splits_dict["splits"]

                rows_args_list = [{"dataset": dataset, "config": config, "split": split} for split in splits]
                local_rows_reports = [get_kwargs_report(get_rows_response, args) for args in rows_args_list]
                rows_reports += local_rows_reports

    endpoints["/datasets"] = get_endpoint_report("/datasets", datasets_reports)
    endpoints["/info"] = get_endpoint_report("/info", info_reports)
    endpoints["/configs"] = get_endpoint_report("/configs", configs_reports)
    endpoints["/splits"] = get_endpoint_report("/splits", splits_reports)
    endpoints["/rows"] = get_endpoint_report("/rows", rows_reports)

    return {"endpoints": endpoints}
