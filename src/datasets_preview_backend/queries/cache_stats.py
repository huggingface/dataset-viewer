from time import time
from typing import Any, Dict, List, TypedDict, Union, cast

from datasets_preview_backend.queries.configs import get_configs
from datasets_preview_backend.queries.datasets import get_datasets
from datasets_preview_backend.queries.infos import get_infos
from datasets_preview_backend.queries.rows import get_rows
from datasets_preview_backend.queries.splits import get_splits
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


def get_kwargs_report(memoized_function: Any, kwargs: Any) -> ArgsCacheStats:
    cache = memoized_function.__cache__
    # cache.close()
    key = memoized_function.__cache_key__(**kwargs)
    content, expire_time = cache.get(key, default=None, expire_time=True)
    is_cached = content is not None
    is_expired = expire_time is not None and expire_time < time()
    is_error = isinstance(content, Exception)
    is_valid = is_cached and not is_expired and not is_error
    return {
        "kwargs": kwargs,
        "is_cached": is_cached,
        "is_expired": is_expired,
        "is_error": is_error,
        "is_valid": is_valid,
        "content": content,
    }


def get_cache_stats() -> CacheStats:
    endpoints = {}

    datasets_kwargs_list: Any = [{}]
    datasets_reports = [get_kwargs_report(get_datasets, kwargs) for kwargs in datasets_kwargs_list]
    infos_reports: List[ArgsCacheStats] = []
    configs_reports: List[ArgsCacheStats] = []
    splits_reports: List[ArgsCacheStats] = []
    rows_reports: List[ArgsCacheStats] = []

    valid_datasets_reports = [d for d in datasets_reports if d["is_valid"]]
    for datasets_report in valid_datasets_reports:
        datasets_content = cast(DatasetsContent, datasets_report["content"])
        datasets = datasets_content["datasets"]

        configs_kwargs_list = [{"dataset": dataset["dataset"]} for dataset in datasets]
        local_configs_reports = [get_kwargs_report(get_configs, kwargs) for kwargs in configs_kwargs_list]
        configs_reports += local_configs_reports

        valid_configs_reports = [d for d in local_configs_reports if d["is_valid"]]
        for configs_report in valid_configs_reports:
            configs_content = cast(ConfigsContent, configs_report["content"])
            configs = configs_content["configs"]

            infos_kwargs_list = [{"dataset": config["dataset"], "config": config["config"]} for config in configs]
            local_infos_reports = [get_kwargs_report(get_infos, kwargs) for kwargs in infos_kwargs_list]
            infos_reports += local_infos_reports

            splits_kwargs_list = [{"dataset": config["dataset"], "config": config["config"]} for config in configs]
            local_splits_reports = [get_kwargs_report(get_splits, kwargs) for kwargs in splits_kwargs_list]
            splits_reports += local_splits_reports

            valid_splits_reports = [d for d in local_splits_reports if d["is_valid"]]
            for splits_report in valid_splits_reports:
                splits_content = cast(SplitsContent, splits_report["content"])
                splits = splits_content["splits"]

                rows_args_list = [
                    {"dataset": split["dataset"], "config": split["config"], "split": split["split"]}
                    for split in splits
                ]
                local_rows_reports = [get_kwargs_report(get_rows, args) for args in rows_args_list]
                rows_reports += local_rows_reports

    endpoints["/datasets"] = get_endpoint_report("/datasets", datasets_reports)
    endpoints["/infos"] = get_endpoint_report("/infos", infos_reports)
    endpoints["/configs"] = get_endpoint_report("/configs", configs_reports)
    endpoints["/splits"] = get_endpoint_report("/splits", splits_reports)
    endpoints["/rows"] = get_endpoint_report("/rows", rows_reports)

    return {"endpoints": endpoints}
