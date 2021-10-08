from typing import List, Optional, TypedDict

from datasets import get_dataset_split_names

from datasets_preview_backend.cache import cache, memoize  # type: ignore
from datasets_preview_backend.constants import DATASETS_BLOCKLIST
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.queries.configs import get_configs


class SplitItem(TypedDict):
    dataset: str
    config: str
    split: str


class SplitsContent(TypedDict):
    splits: List[SplitItem]


def get_split_items(dataset: str, config: str) -> List[SplitItem]:
    try:
        splits = get_dataset_split_names(dataset, config)
    except FileNotFoundError as err:
        raise Status404Error("The dataset config could not be found.", err)
    except ValueError as err:
        if str(err).startswith(f"BuilderConfig {config} not found."):
            raise Status404Error("The dataset config could not be found.", err)
        else:
            raise Status400Error("The split names could not be parsed from the dataset config.", err)
    except Exception as err:
        raise Status400Error("The split names could not be parsed from the dataset config.", err)
    return [{"dataset": dataset, "config": config, "split": split} for split in splits]


@memoize(cache)  # type:ignore
def get_splits(*, dataset: str, config: Optional[str] = None) -> SplitsContent:
    if not isinstance(dataset, str) and dataset is not None:
        raise TypeError("dataset argument should be a string")
    if dataset is None:
        raise Status400Error("'dataset' is a required query parameter.")
    if dataset in DATASETS_BLOCKLIST:
        raise Status400Error("this dataset is not supported for now.")
    if config is not None and not isinstance(config, str):
        raise TypeError("config argument should be a string")

    if config is None:
        # recurse to get cached entries
        split_items = [
            split_item
            for config_item in get_configs(dataset=dataset)["configs"]
            for split_item in get_splits(dataset=dataset, config=config_item["config"])["splits"]
        ]
    else:
        split_items = get_split_items(dataset, config)

    return {"splits": split_items}
