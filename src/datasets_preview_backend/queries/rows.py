import logging
import re
from typing import Any, List, Optional, TypedDict

from datasets import IterableDataset, load_dataset

from datasets_preview_backend.cache import cache, memoize  # type: ignore
from datasets_preview_backend.config import EXTRACT_ROWS_LIMIT
from datasets_preview_backend.constants import DATASETS_BLOCKLIST
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.queries.configs import get_configs
from datasets_preview_backend.queries.infos import get_infos
from datasets_preview_backend.queries.splits import get_splits

logger = logging.getLogger(__name__)


class Feature(TypedDict):
    name: str
    content: Any


class FeatureItem(TypedDict):
    dataset: str
    config: str
    feature: Feature


class RowItem(TypedDict):
    dataset: str
    config: str
    split: str
    row: Any


class RowsContent(TypedDict):
    features: List[FeatureItem]
    rows: List[RowItem]


def get_row_items(dataset: str, config: str, split: str) -> List[RowItem]:
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

    return [{"dataset": dataset, "config": config, "split": split, "row": row} for row in rows]


def get_feature_items(dataset: str, config: str) -> List[FeatureItem]:
    # note: the function might raise
    infos_content = get_infos(dataset=dataset, config=config)
    infoItems = [infoItem["info"] for infoItem in infos_content["infos"]]

    if len(infoItems) != 1:
        raise Exception("a dataset config should have exactly one info")
    infoItem = infoItems[0]
    if "features" not in infoItem or infoItem["features"] is None:
        raise Status400Error("a dataset config info should contain a 'features' property")
    return [
        {"dataset": dataset, "config": config, "feature": {"name": name, "content": content}}
        for (name, content) in infoItem["features"].items()
    ]


@memoize(cache)  # type:ignore
def get_rows(*, dataset: str, config: Optional[str] = None, split: Optional[str] = None) -> RowsContent:
    if not isinstance(dataset, str) and dataset is not None:
        raise TypeError("dataset argument should be a string")
    if dataset is None:
        raise Status400Error("'dataset' is a required query parameter.")
    if dataset in DATASETS_BLOCKLIST:
        raise Status400Error("this dataset is not supported for now.")
    if config is not None and not isinstance(config, str):
        raise TypeError("config argument should be a string")
    if not isinstance(split, str) and split is not None:
        raise TypeError("split argument should be a string")

    if config is None:
        # split is ignored if config is not passed
        logger.debug("split argument is ignored since config is not provided")
        split = None

    if config is None:
        # recurse to get cached entries
        row_items = [
            row_item
            for config_item in get_configs(dataset=dataset)["configs"]
            for row_item in get_rows(dataset=dataset, config=config_item["config"])["rows"]
        ]
    elif split is None:
        # recurse to get cached entries
        row_items = [
            row_item
            for split_item in get_splits(dataset=dataset, config=config)["splits"]
            for row_item in get_rows(dataset=dataset, config=config, split=split_item["split"])["rows"]
        ]
    else:
        row_items = get_row_items(dataset, config, split)

    if config is None:
        # recurse to get cached entries
        feature_items = [
            feature_item
            for config_item in get_configs(dataset=dataset)["configs"]
            for feature_item in get_rows(dataset=dataset, config=config_item["config"])["features"]
        ]
    else:
        feature_items = get_feature_items(dataset, config)

    return {"features": feature_items, "rows": row_items}
