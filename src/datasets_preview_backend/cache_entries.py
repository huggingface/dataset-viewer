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

memoized_functions = {
    "/datasets": get_datasets,
    "/configs": get_configs,
    "/infos": get_infos,
    "/splits": get_splits,
    "/rows": get_rows,
}


class CacheEntry(TypedDict):
    endpoint: str
    kwargs: Dict[str, Union[str, int]]
    status: str
    content: Union[Content, None]
    error: Union[Exception, None]


def get_cache_entry(endpoint: str, kwargs: Any) -> CacheEntry:
    memoized_function = memoized_functions[endpoint]
    cache = memoized_function.__cache__
    key = memoized_function.__cache_key__(**kwargs)
    cache_content = cache.get(key, default=None)

    error: Union[Exception, None] = None
    content: Union[Content, None] = None
    if isinstance(cache_content, Exception):
        error = cache_content
    else:
        content = cache_content
    # we only report the cached datasets as valid
    # as we rely on cache warming at startup (otherwise, the first call would take too long - various hours)
    # note that warming can be done by 1. calling /datasets, then 2. calling /rows?dataset={dataset}
    # for all the datasets
    # TODO: use an Enum?
    status = "cache_miss" if cache_content is None else "error" if error is not None else "valid"
    return {
        "endpoint": endpoint,
        "kwargs": kwargs,
        "status": status,
        "content": content,
        "error": error,
    }


def get_cache_entries() -> List[CacheEntry]:
    cache_entries: List[CacheEntry] = []

    datasets_kwargs_list: Any = [{}]
    local_datasets_entries = [get_cache_entry(endpoint="/datasets", kwargs=kwargs) for kwargs in datasets_kwargs_list]
    cache_entries += local_datasets_entries

    valid_datasets_entries = [d for d in local_datasets_entries if d["status"] == "valid"]
    for datasets_report in valid_datasets_entries:
        datasets_content = cast(DatasetsContent, datasets_report["content"])
        datasets = datasets_content["datasets"]

        configs_kwargs_list = [{"dataset": dataset["dataset"]} for dataset in datasets]
        local_configs_entries = [get_cache_entry(endpoint="/configs", kwargs=kwargs) for kwargs in configs_kwargs_list]
        cache_entries += local_configs_entries

        valid_configs_entries = [d for d in local_configs_entries if d["status"] == "valid"]
        for configs_report in valid_configs_entries:
            configs_content = cast(ConfigsContent, configs_report["content"])
            configs = configs_content["configs"]

            infos_kwargs_list = [{"dataset": config["dataset"], "config": config["config"]} for config in configs]
            local_infos_entries = [get_cache_entry(endpoint="/infos", kwargs=kwargs) for kwargs in infos_kwargs_list]
            cache_entries += local_infos_entries

            splits_kwargs_list = [{"dataset": config["dataset"], "config": config["config"]} for config in configs]
            local_splits_entries = [
                get_cache_entry(endpoint="/splits", kwargs=kwargs) for kwargs in splits_kwargs_list
            ]
            cache_entries += local_splits_entries

            valid_splits_entries = [d for d in local_splits_entries if d["status"] == "valid"]
            for splits_report in valid_splits_entries:
                splits_content = cast(SplitsContent, splits_report["content"])
                splits = splits_content["splits"]

                rows_kwargs_list = [
                    {"dataset": split["dataset"], "config": split["config"], "split": split["split"]}
                    for split in splits
                ]
                local_rows_entries = [get_cache_entry(endpoint="/rows", kwargs=kwargs) for kwargs in rows_kwargs_list]
                cache_entries += local_rows_entries
    return cache_entries
