from typing import Any, Dict, List, TypedDict, Union, cast

from diskcache.core import ENOVAL  # type: ignore

from datasets_preview_backend.exceptions import StatusError, StatusErrorContent
from datasets_preview_backend.queries.configs import (
    ConfigItem,
    ConfigsContent,
    get_configs,
)
from datasets_preview_backend.queries.datasets import DatasetItem, get_datasets
from datasets_preview_backend.queries.infos import InfosContent, get_infos
from datasets_preview_backend.queries.rows import RowsContent, get_rows
from datasets_preview_backend.queries.splits import SplitItem, SplitsContent, get_splits

Content = Union[ConfigsContent, InfosContent, RowsContent, SplitsContent]


class CacheEntry(TypedDict):
    endpoint: str
    kwargs: Dict[str, str]
    status: str
    content: Union[Content, None]
    error: Union[StatusErrorContent, None]


def get_cache_entry(endpoint: str, func: Any, kwargs: Any) -> CacheEntry:
    try:
        cache_content = func(**kwargs, _lookup=True)
        if cache_content == ENOVAL:
            status = "cache_miss"
            content = None
        else:
            status = "valid"
            content = cache_content
        error = None
    except StatusError as err:
        status = "error"
        content = None
        error = err.as_content()
    except Exception:
        status = "server error"
        content = None
        error = None

    return {
        "endpoint": endpoint,
        "kwargs": kwargs,
        "status": status,
        "content": content,
        "error": error,
    }


def get_expected_split_entries(split_item: SplitItem) -> List[CacheEntry]:
    # Note: SplitItem has the same form as get_rows kwargs

    rows_cache_entry = get_cache_entry("/rows", get_rows, split_item)
    return [rows_cache_entry]


def get_expected_config_entries(config_item: ConfigItem) -> List[CacheEntry]:
    # Note: ConfigItem has the same form as get_splits and get_infos kwargs

    infos_cache_entry = get_cache_entry("/infos", get_infos, config_item)
    entries = [infos_cache_entry]

    splits_cache_entry = get_cache_entry("/splits", get_splits, config_item)
    entries += [splits_cache_entry]

    if splits_cache_entry["status"] == "valid":
        split_items = cast(SplitsContent, splits_cache_entry["content"])["splits"]
        sub_entries = [
            cache_entry for split_item in split_items for cache_entry in get_expected_split_entries(split_item)
        ]
        entries += sub_entries

    return entries


def get_expected_dataset_entries(dataset_item: DatasetItem) -> List[CacheEntry]:
    # Note: DatasetItem has the same form as get_configs kwargs

    configs_cache_entry = get_cache_entry("/configs", get_configs, dataset_item)
    entries = [configs_cache_entry]

    if configs_cache_entry["status"] == "valid":
        config_items = cast(ConfigsContent, configs_cache_entry["content"])["configs"]
        sub_entries = [
            cache_entry for config_item in config_items for cache_entry in get_expected_config_entries(config_item)
        ]
        entries += sub_entries

    return entries


def get_expected_entries() -> List[CacheEntry]:
    dataset_items = get_datasets()["datasets"]

    return [
        cache_entry for dataset_item in dataset_items for cache_entry in get_expected_dataset_entries(dataset_item)
    ]
