import logging
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, TypedDict, Union

import numpy  # type: ignore
from datasets import (
    IterableDataset,
    get_dataset_config_names,
    get_dataset_split_names,
    list_datasets,
    load_dataset,
    load_dataset_builder,
)
from datasets.utils.download_manager import GenerateMode  # type: ignore
from PIL import Image  # type: ignore

from datasets_preview_backend.assets import create_asset_file, create_image_file
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


Cell = Any


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


class CellTypeError(Exception):
    pass


class FeatureTypeError(Exception):
    pass


def generate_image_url_cell(dataset: str, config: str, split: str, row_idx: int, column: str, cell: Cell) -> Cell:
    # TODO: also manage nested dicts and other names
    if column != "image_url":
        raise CellTypeError("image URL column must be named 'image_url'")
    if type(cell) != str:
        raise CellTypeError("image URL column must be a string")
    return cell


def generate_image_cell(dataset: str, config: str, split: str, row_idx: int, column: str, cell: Cell) -> Cell:
    if column != "image":
        raise CellTypeError("image column must be named 'image'")
    try:
        filename = cell["filename"]
        data = cell["data"]
    except Exception:
        raise CellTypeError("image cell must contain 'filename' and 'data' fields")
    if type(filename) != str:
        raise CellTypeError("'filename' field must be a string")
    if type(data) != bytes:
        raise CellTypeError("'data' field must be a bytes")

    # this function can raise, we don't catch it
    return create_asset_file(dataset, config, split, row_idx, column, filename, data)


def generate_array2d_image_cell(dataset: str, config: str, split: str, row_idx: int, column: str, cell: Cell) -> Cell:
    if column != "image":
        raise CellTypeError("image column must be named 'image'")
    if (
        not isinstance(cell, list)
        or len(cell) == 0
        or not isinstance(cell[0], list)
        or len(cell[0]) == 0
        or type(cell[0][0]) != int
    ):
        raise CellTypeError("array2d image cell must contain 2D array of integers")
    array = 255 - numpy.asarray(cell, dtype=numpy.uint8)
    mode = "L"
    image = Image.fromarray(array, mode)
    filename = "image.jpg"

    return create_image_file(dataset, config, split, row_idx, column, filename, image)


# TODO: use the features to help generating the cells?
cell_generators = [generate_image_cell, generate_array2d_image_cell, generate_image_url_cell]


def generate_cell(dataset: str, config: str, split: str, row_idx: int, column: str, cell: Cell) -> Cell:
    for cell_generator in cell_generators:
        try:
            return cell_generator(dataset, config, split, row_idx, column, cell)
        except CellTypeError:
            pass
    return cell


def get_cells(row: Row) -> List[Cell]:
    try:
        return list(row.items())
    except Exception as err:
        raise Status400Error("cells could not be extracted from the row", err)


def generate_row(dataset: str, config: str, split: str, row: Row, row_idx: int) -> Row:
    return {column: generate_cell(dataset, config, split, row_idx, column, cell) for (column, cell) in get_cells(row)}


def check_feature_type(value: Any, type: str, dtype: str) -> None:
    if "_type" not in value or value["_type"] != type:
        raise TypeError("_type is not the expected value")
    if "dtype" not in value or value["dtype"] != dtype:
        raise TypeError("dtype is not the expected value")


def generate_image_url_feature(name: str, content: Any) -> Any:
    if name != "image_url":
        raise FeatureTypeError("image URL column must be named 'image_url'")
    try:
        check_feature_type(content, "Value", "string")
    except Exception:
        raise FeatureTypeError("image URL feature must be a string")
    # Custom "_type": "ImageFile"
    return {"id": None, "_type": "ImageUrl"}


def generate_image_feature(name: str, content: Any) -> Any:
    if name != "image":
        raise FeatureTypeError("image column must be named 'image'")
    try:
        check_feature_type(content["filename"], "Value", "string")
        check_feature_type(content["data"], "Value", "binary")
    except Exception:
        raise FeatureTypeError("image feature must contain 'filename' and 'data' fields")
    # Custom "_type": "ImageFile"
    return {"id": None, "_type": "ImageFile"}


def generate_array2d_image_feature(name: str, content: Any) -> Any:
    if name != "image":
        raise FeatureTypeError("image column must be named 'image'")
    try:
        check_feature_type(content, "Array2D", "uint8")
    except Exception:
        raise FeatureTypeError("array2D image feature must have type uint8")
    # we also have shape in the feature: shape: [28, 28] for MNIST
    # Custom "_type": "ImageFile"
    return {"id": None, "_type": "ImageFile"}


feature_generators = [generate_image_feature, generate_array2d_image_feature, generate_image_url_feature]


def generate_feature_content(column: str, content: Any) -> Any:
    for feature_generator in feature_generators:
        try:
            return feature_generator(column, content)
        except FeatureTypeError:
            pass
    return content


def get_rows(dataset: str, config: str, split: str) -> List[Row]:
    num_rows = EXTRACT_ROWS_LIMIT

    try:
        iterable_dataset = load_dataset(
            dataset, name=config, split=split, streaming=True, download_mode=GenerateMode.FORCE_REDOWNLOAD
        )
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

    return [generate_row(dataset, config, split, row, row_idx) for row_idx, row in enumerate(rows)]


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
        split_names: List[str] = get_dataset_split_names(dataset, config, download_mode=GenerateMode.FORCE_REDOWNLOAD)
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
        builder = load_dataset_builder(dataset, name=config, download_mode=GenerateMode.FORCE_REDOWNLOAD)
        info = asdict(builder.info)
        if "splits" in info and info["splits"] is not None:
            info["splits"] = {split_name: split_info for split_name, split_info in info["splits"].items()}
    except FileNotFoundError as err:
        raise Status404Error("The config info could not be found.", err)
    except Exception as err:
        raise Status400Error("The config info could not be parsed from the dataset.", err)
    return info


def get_features(info: Info) -> List[Feature]:
    try:
        features = [] if info["features"] is None else info["features"].items()
        return [{"name": name, "content": generate_feature_content(name, content)} for (name, content) in features]
    except Exception as err:
        # note that no exception will be raised if features exists, but is empty
        raise Status400Error("features not found in dataset config info", err)


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
        config_names: List[str] = get_dataset_config_names(dataset, download_mode=GenerateMode.FORCE_REDOWNLOAD)
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


def delete_dataset_entry(dataset: str) -> None:
    get_dataset_entry(dataset=dataset, _delete=True)


@memoize(cache)  # type:ignore
def get_dataset_names() -> List[str]:
    # If an exception is raised, we let it propagate
    return list_datasets(with_community_datasets=True, with_details=False)  # type: ignore


def get_refreshed_dataset_names() -> List[str]:
    return get_dataset_names(_refresh=True)  # type: ignore
