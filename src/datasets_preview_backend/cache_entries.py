from typing import Any, Dict, List, TypedDict, Union, cast

from diskcache.core import ENOVAL  # type: ignore

from datasets_preview_backend.exceptions import StatusErrorContent
from datasets_preview_backend.queries.configs import ConfigsContent, get_configs
from datasets_preview_backend.queries.datasets import DatasetsContent, get_datasets
from datasets_preview_backend.queries.infos import InfosContent, get_infos
from datasets_preview_backend.queries.rows import RowsContent, get_rows
from datasets_preview_backend.queries.splits import SplitsContent, get_splits

Content = Union[
    ConfigsContent,
    DatasetsContent,
    InfosContent,
    RowsContent,
    SplitsContent,
    StatusErrorContent,
]


class CacheEntry(TypedDict):
    endpoint: str
    kwargs: Dict[str, Union[str, int]]
    status: str
    content: Union[Content, None]
    error: Union[Exception, None]


def get_cache_entry(endpoint: str, func: Any, kwargs: Any) -> CacheEntry:
    try:
        content = func(**kwargs, _lookup=True)
    except Exception as err:
        content = err

    is_cache_miss = content == ENOVAL
    is_error = isinstance(content, Exception)

    return {
        "endpoint": endpoint,
        "kwargs": kwargs,
        "status": "cache_miss" if is_cache_miss else "error" if is_error else "valid",
        "content": None if is_error else content,
        "error": content if is_error else None,
    }


def get_cache_entries() -> List[CacheEntry]:
    cache_entries: List[CacheEntry] = []

    datasets_kwargs_list: Any = [{}]
    local_datasets_entries = [get_cache_entry("/datasets", get_datasets, kwargs) for kwargs in datasets_kwargs_list]
    cache_entries += local_datasets_entries

    valid_datasets_entries = [d for d in local_datasets_entries if d["status"] == "valid"]
    for datasets_report in valid_datasets_entries:
        datasets_content = cast(DatasetsContent, datasets_report["content"])
        datasets = datasets_content["datasets"]

        configs_kwargs_list = [{"dataset": dataset["dataset"]} for dataset in datasets]
        local_configs_entries = [get_cache_entry("/configs", get_configs, kwargs) for kwargs in configs_kwargs_list]
        cache_entries += local_configs_entries

        valid_configs_entries = [d for d in local_configs_entries if d["status"] == "valid"]
        for configs_report in valid_configs_entries:
            configs_content = cast(ConfigsContent, configs_report["content"])
            configs = configs_content["configs"]

            infos_kwargs_list = [{"dataset": config["dataset"], "config": config["config"]} for config in configs]
            local_infos_entries = [get_cache_entry("/infos", get_infos, kwargs) for kwargs in infos_kwargs_list]
            cache_entries += local_infos_entries

            splits_kwargs_list = [{"dataset": config["dataset"], "config": config["config"]} for config in configs]
            local_splits_entries = [get_cache_entry("/splits", get_splits, kwargs) for kwargs in splits_kwargs_list]
            cache_entries += local_splits_entries

            valid_splits_entries = [d for d in local_splits_entries if d["status"] == "valid"]
            for splits_report in valid_splits_entries:
                splits_content = cast(SplitsContent, splits_report["content"])
                splits = splits_content["splits"]

                rows_kwargs_list = [
                    {"dataset": split["dataset"], "config": split["config"], "split": split["split"]}
                    for split in splits
                ]
                local_rows_entries = [get_cache_entry("/rows", get_rows, kwargs) for kwargs in rows_kwargs_list]
                cache_entries += local_rows_entries
    return cache_entries
