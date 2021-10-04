from time import time
from typing import Any, Dict, List, TypedDict, Union, cast

from datasets_preview_backend.responses import memoized_functions

from datasets_preview_backend.types import (
    ConfigsContent,
    Content,
    DatasetsContent,
    SplitsContent,
)


class ArgsCacheStats(TypedDict):
    kwargs: Dict[str, Union[str, int]]
    is_cached: bool
    is_expired: bool
    is_error: bool
    is_valid: bool
    content: Union[Content, None]


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


def get_kwargs_report(endpoint: str, kwargs: Any) -> ArgsCacheStats:
    memoized_function = memoized_functions[endpoint]
    cache = memoized_function.__cache__
    # cache.close()
    key = memoized_function.__cache_key__(**kwargs)
    content, expire_time = cache.get(key, default=None, expire_time=True)
    is_cached = content is not None
    is_expired = expire_time is not None and expire_time < time()
    is_error = isinstance(content, Exception)
    is_valid = is_cached and not is_expired and not is_error
    return {
        "endpoint": endpoint,
        "kwargs": kwargs,
        "is_cached": is_cached,
        "is_expired": is_expired,
        "is_error": is_error,
        "is_valid": is_valid,
        "content": content,
    }


def get_cache_reports() -> List[ArgsCacheStats]:
    reports: List[ArgsCacheStats] = []

    datasets_kwargs_list: Any = [{}]
    local_datasets_reports = [
        get_kwargs_report(endpoint="/datasets", kwargs=kwargs) for kwargs in datasets_kwargs_list
    ]

    valid_datasets_reports = [d for d in local_datasets_reports if d["is_valid"]]
    for datasets_report in valid_datasets_reports:
        datasets_content = cast(DatasetsContent, datasets_report["content"])
        datasets = datasets_content["datasets"]

        configs_kwargs_list = [{"dataset": dataset["dataset"]} for dataset in datasets]
        local_configs_reports = [
            get_kwargs_report(endpoint="/configs", kwargs=kwargs) for kwargs in configs_kwargs_list
        ]
        reports += local_configs_reports

        valid_configs_reports = [d for d in local_configs_reports if d["is_valid"]]
        for configs_report in valid_configs_reports:
            configs_content = cast(ConfigsContent, configs_report["content"])
            configs = configs_content["configs"]

            infos_kwargs_list = [{"dataset": config["dataset"], "config": config["config"]} for config in configs]
            local_infos_reports = [get_kwargs_report(endpoint="/infos", kwargs=kwargs) for kwargs in infos_kwargs_list]
            reports += local_infos_reports

            splits_kwargs_list = [{"dataset": config["dataset"], "config": config["config"]} for config in configs]
            local_splits_reports = [
                get_kwargs_report(endpoint="/splits", kwargs=kwargs) for kwargs in splits_kwargs_list
            ]
            reports += local_splits_reports

            valid_splits_reports = [d for d in local_splits_reports if d["is_valid"]]
            for splits_report in valid_splits_reports:
                splits_content = cast(SplitsContent, splits_report["content"])
                splits = splits_content["splits"]

                rows_kwargs_list = [
                    {"dataset": split["dataset"], "config": split["config"], "split": split["split"]}
                    for split in splits
                ]
                local_rows_reports = [
                    get_kwargs_report(endpoint="/rows", kwargs=kwargs) for kwargs in rows_kwargs_list
                ]
                reports += local_rows_reports
    return reports


def get_cache_stats() -> CacheStats:
    reports = get_cache_reports()

    endpoints = {
        endpoint: get_endpoint_report(endpoint, [report for report in reports if report["endpoint"] == endpoint])
        for endpoint in memoized_functions
    }

    return {"endpoints": endpoints}
