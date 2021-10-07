from time import time
from typing import Any, Dict, List, TypedDict, Union, cast

from datasets_preview_backend.config import CACHE_SHORT_TTL_SECONDS
from datasets_preview_backend.responses import memoized_functions
from datasets_preview_backend.types import (
    ConfigsContent,
    Content,
    DatasetsContent,
    SplitsContent,
)


class ArgsCacheStats(TypedDict):
    endpoint: str
    kwargs: Dict[str, Union[str, int]]
    status: str
    content: Union[Content, None]


def get_kwargs_report(endpoint: str, kwargs: Any) -> ArgsCacheStats:
    memoized_function = memoized_functions[endpoint]
    cache = memoized_function.__cache__
    key = memoized_function.__cache_key__(**kwargs)
    content, expire_time = cache.get(key, default=None, expire_time=True)
    # we only report the cached datasets as valid
    # as we rely on cache warming at startup (otherwise, the first call would take too long - various hours)
    # note that warming can be done by 1. calling /datasets, then 2. calling /rows?dataset={dataset}
    # for all the datasets
    # TODO: use an Enum?
    status = (
        "cache_miss"
        if content is None
        else "cache_expired"
        if expire_time is not None and expire_time < time()
        else "error"
        if isinstance(content, Exception)
        else "valid"
    )
    return {
        "endpoint": endpoint,
        "kwargs": kwargs,
        "status": status,
        "content": content,
    }


@memoize(cache=cache, expire=CACHE_SHORT_TTL_SECONDS)  # type:ignore
def get_cache_reports() -> List[ArgsCacheStats]:
    reports: List[ArgsCacheStats] = []

    datasets_kwargs_list: Any = [{}]
    local_datasets_reports = [
        get_kwargs_report(endpoint="/datasets", kwargs=kwargs) for kwargs in datasets_kwargs_list
    ]
    reports += local_datasets_reports

    valid_datasets_reports = [d for d in local_datasets_reports if d["status"] == "valid"]
    for datasets_report in valid_datasets_reports:
        datasets_content = cast(DatasetsContent, datasets_report["content"])
        datasets = datasets_content["datasets"]

        configs_kwargs_list = [{"dataset": dataset["dataset"]} for dataset in datasets]
        local_configs_reports = [
            get_kwargs_report(endpoint="/configs", kwargs=kwargs) for kwargs in configs_kwargs_list
        ]
        reports += local_configs_reports

        valid_configs_reports = [d for d in local_configs_reports if d["status"] == "valid"]
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

            valid_splits_reports = [d for d in local_splits_reports if d["status"] == "valid"]
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
