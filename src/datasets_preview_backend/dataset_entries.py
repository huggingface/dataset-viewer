import logging
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, TypedDict, Union

from datasets import (
    IterableDataset,
    get_dataset_config_names,
    get_dataset_split_names,
    list_datasets,
    load_dataset,
    load_dataset_builder,
)

from datasets_preview_backend.cache import (  # type: ignore
    CacheNotFoundError,
    cache,
    memoize,
)
from datasets_preview_backend.config import EXTRACT_ROWS_LIMIT
from datasets_preview_backend.constants import DATASETS_BLOCKLIST, DEFAULT_CONFIG_NAME
from datasets_preview_backend.exceptions import (
    Status400Error,
    Status404Error,
    StatusError,
    StatusErrorContent,
)

logger = logging.getLogger(__name__)


class Feature(TypedDict):
    name: str
    content: Any


Row = Any


class Split(TypedDict):
    split: str
    rows: List[Row]


Info = Dict[str, Any]


class Config(TypedDict):
    config: str
    splits: List[Split]
    info: Info
    features: List[Feature]


class DatasetEntry(TypedDict):
    dataset: str
    configs: List[Config]


def get_rows(dataset: str, config: str, split: str) -> List[Row]:
    num_rows = EXTRACT_ROWS_LIMIT

    try:
        iterable_dataset = load_dataset(dataset, name=config, split=split, streaming=True)
        if not isinstance(iterable_dataset, IterableDataset):
            raise TypeError("load_dataset should return an IterableDataset")
        rows = list(iterable_dataset.take(num_rows))
    except FileNotFoundError as err:
        raise Status404Error("The split for the dataset config could not be found.", err)
    except NotImplementedError as err:
        # TODO: check what has changed once https://github.com/huggingface/datasets/pull/2662 is merged
        try:
            regex = re.compile(r"Extraction protocol for file at .*?((\.\w+)?\.\w+)* is not implemented yet")
            match = regex.match(str(err))
            if match is None:
                raise Exception("No match")
            extension = match.group(1)
        except Exception:
            raise Status400Error("The rows could not be extracted from the split of the dataset config.", err)
        else:
            raise Status400Error(
                "The rows could not be extracted from the split of the dataset config because extension"
                f" {extension} is not supported.",
                err,
            )
    except ValueError as err:
        if (
            str(err).startswith(f"BuilderConfig {config} not found.")
            or str(err).startswith("Config name is missing.")
            or str(err).startswith("Bad split")
        ):
            raise Status404Error("The dataset config could not be found.", err)
        else:
            raise Status400Error("The rows could not be extracted from the split of the dataset config.", err)
    except Exception as err:
        raise Status400Error("The rows could not be extracted from the split of the dataset config.", err)

    if len(rows) != num_rows:
        logger.info(
            f"could not read all the required rows ({len(rows)} / {num_rows}) from dataset {dataset} -"
            f" {config} - {split}"
        )

    return rows


def filter_split_entries(split_entries: List[Split], split: Optional[str] = None) -> List[Split]:
    if split is not None:
        if not isinstance(split, str):
            raise TypeError("split argument should be a string")
        split_entries = [split_entry for split_entry in split_entries if split_entry["split"] == split]
        if not split_entries:
            raise Status404Error("split not found in config")
    return split_entries


def get_split(dataset: str, config: str, split: str) -> Split:
    rows = get_rows(dataset, config, split)
    return {"split": split, "rows": rows}


def get_split_names(dataset: str, config: str) -> List[str]:
    try:
        split_names: List[str] = get_dataset_split_names(dataset, config)
    except FileNotFoundError as err:
        raise Status404Error("The dataset config could not be found.", err)
    except ValueError as err:
        if str(err).startswith(f"BuilderConfig {config} not found."):
            raise Status404Error("The dataset config could not be found.", err)
        else:
            raise Status400Error("The split names could not be parsed from the dataset config.", err)
    except Exception as err:
        raise Status400Error("The split names could not be parsed from the dataset config.", err)

    return split_names


def get_info(dataset: str, config: str) -> Info:
    try:
        # TODO: use get_dataset_infos if https://github.com/huggingface/datasets/issues/3013 is fixed
        builder = load_dataset_builder(dataset, name=config)
        info = asdict(builder.info)
        if "splits" in info and info["splits"] is not None:
            info["splits"] = {split_name: split_info for split_name, split_info in info["splits"].items()}
    except FileNotFoundError as err:
        raise Status404Error("The config info could not be found.", err)
    except Exception as err:
        raise Status400Error("The config info could not be parsed from the dataset.", err)
    return info


def get_features(info: Info) -> List[Feature]:
    if "features" not in info or info["features"] is None:
        raise Status400Error("a dataset config info should contain a 'features' property")
    return [{"name": name, "content": content} for (name, content) in info["features"].items()]


def get_config_name(config_entry: Config) -> str:
    return config_entry["config"]


def filter_config_entries(config_entries: List[Config], config: Optional[str] = None) -> List[Config]:
    if config is not None:
        if not isinstance(config, str):
            raise TypeError("config argument should be a string")
        config_entries = [config_entry for config_entry in config_entries if config_entry["config"] == config]
        if not config_entries:
            raise Status404Error("config not found in dataset")
    return config_entries


def get_config(dataset: str, config_name: str) -> Config:
    if not isinstance(config_name, str):
        raise TypeError("config_name argument should be a string")
    split_names = get_split_names(dataset, config_name)
    splits = [get_split(dataset, config_name, split_name) for split_name in split_names]
    info = get_info(dataset, config_name)
    features = get_features(info)

    return {"config": config_name, "splits": splits, "info": info, "features": features}


def get_config_names(dataset: str) -> List[str]:
    try:
        config_names: List[str] = get_dataset_config_names(dataset)
        if not config_names:
            config_names = [DEFAULT_CONFIG_NAME]
    except FileNotFoundError as err:
        raise Status404Error("The dataset could not be found.", err)
    except Exception as err:
        raise Status400Error("The config names could not be parsed from the dataset.", err)

    return config_names


class DatasetCacheStatus(TypedDict):
    dataset: str
    status: str
    content: Union[DatasetEntry, None]
    error: Union[StatusErrorContent, None]


def get_dataset_cache_status(dataset: str) -> DatasetCacheStatus:
    try:
        cache_content = get_dataset_entry(dataset=dataset, _lookup=True)
        status = "valid"
        content = cache_content
        error = None
    except CacheNotFoundError:
        status = "cache_miss"
        content = None
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
        "dataset": dataset,
        "status": status,
        "content": content,
        "error": error,
    }


@memoize(cache)  # type:ignore
def get_dataset_entry(*, dataset: str) -> DatasetEntry:
    if not isinstance(dataset, str) and dataset is not None:
        raise TypeError("dataset argument should be a string")
    if dataset is None:
        raise Status400Error("'dataset' is a required query parameter.")
    if dataset in DATASETS_BLOCKLIST:
        raise Status400Error("this dataset is not supported for now.")

    config_names = get_config_names(dataset)
    configs = [get_config(dataset, config_name) for config_name in config_names]

    return {"dataset": dataset, "configs": configs}


def get_refreshed_dataset_entry(dataset: str) -> DatasetEntry:
    return get_dataset_entry(dataset=dataset, _refresh=True)  # type: ignore


@memoize(cache)  # type:ignore
def get_dataset_names() -> List[str]:
    # If an exception is raised, we let it propagate
    return list_datasets(with_community_datasets=True, with_details=False)  # type: ignore


def get_refreshed_dataset_names() -> List[str]:
    return get_dataset_names(_refresh=True)  # type: ignore
