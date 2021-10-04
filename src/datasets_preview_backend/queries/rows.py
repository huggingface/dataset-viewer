import logging
import re
from typing import List, Optional, cast

from datasets import IterableDataset, load_dataset

from datasets_preview_backend.cache import memoize  # type: ignore
from datasets_preview_backend.config import CACHE_TTL_SECONDS, EXTRACT_ROWS_LIMIT, cache
from datasets_preview_backend.constants import DATASETS_BLOCKLIST
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.queries.configs import get_configs_response
from datasets_preview_backend.queries.infos import get_infos_response
from datasets_preview_backend.queries.splits import get_splits_response
from datasets_preview_backend.responses import CachedResponse
from datasets_preview_backend.types import (
    ConfigsContent,
    FeatureItem,
    InfosContent,
    RowItem,
    RowsContent,
    SplitsContent,
    StatusErrorContent,
)

logger = logging.getLogger(__name__)


def get_rows(dataset: str, config: Optional[str] = None, split: Optional[str] = None) -> RowsContent:
    if not isinstance(dataset, str) and dataset is not None:
        raise TypeError("dataset argument should be a string")
    if dataset is None:
        raise Status400Error("'dataset' is a required query parameter.")
    if dataset in DATASETS_BLOCKLIST:
        raise Status400Error("this dataset is not supported for now.")
    if config is None:
        # split is ignored if config is not passed
        logger.warning("split argument is ignored since config is not provided")
        split = None
    elif not isinstance(config, str):
        raise TypeError("config argument should be a string")
    if not isinstance(split, str) and split is not None:
        raise TypeError("split argument should be a string")
    num_rows = EXTRACT_ROWS_LIMIT

    rowItems: List[RowItem] = []
    featureItems: List[FeatureItem] = []

    if config is not None and split is not None:
        try:
            iterable_dataset = load_dataset(dataset, name=config, split=split, streaming=True)
            if not isinstance(iterable_dataset, IterableDataset):
                raise TypeError("load_dataset should return an IterableDataset")
            rows = list(iterable_dataset.take(num_rows))
        except FileNotFoundError as err:
            raise Status404Error("The split for the dataset config could not be found.") from err
        except NotImplementedError as err:
            # TODO: check what has changed once https://github.com/huggingface/datasets/pull/2662 is merged
            try:
                regex = re.compile(r"Extraction protocol for file at .*?((\.\w+)?\.\w+)* is not implemented yet")
                match = regex.match(str(err))
                if match is None:
                    raise Exception("No match")
                extension = match.group(1)
            except Exception:
                raise Status400Error("The rows could not be extracted from the split of the dataset config.") from err
            else:
                raise Status400Error(
                    "The rows could not be extracted from the split of the dataset config because extension"
                    f" {extension} is not supported."
                ) from err
        except ValueError as err:
            if (
                str(err).startswith(f"BuilderConfig {config} not found.")
                or str(err).startswith("Config name is missing.")
                or str(err).startswith("Bad split")
            ):
                raise Status404Error("The dataset config could not be found.") from err
            else:
                raise Status400Error("The rows could not be extracted from the split of the dataset config.") from err
        except Exception as err:
            raise Status400Error("The rows could not be extracted from the split of the dataset config.") from err

        if len(rows) != num_rows:
            logger.warning(
                f"could not read all the required rows ({len(rows)} / {num_rows}) from dataset {dataset} -"
                f" {config} - {split}"
            )

        rowItems = [{"dataset": dataset, "config": config, "split": split, "row": row} for row in rows]

        content = get_infos_response(dataset=dataset, config=config).content
        if "infos" not in content:
            error = cast(StatusErrorContent, content)
            if "status_code" in error and error["status_code"] == 404:
                raise Status404Error("features could not be found")
            raise Status400Error("features could not be found")
        infos_content = cast(InfosContent, content)
        infoItems = [infoItem["info"] for infoItem in infos_content["infos"]]

        if len(infoItems) != 1:
            raise Exception("a dataset config should have exactly one info")
        infoItem = infoItems[0]
        if "features" not in infoItem:
            raise Status400Error("a dataset config info should contain a 'features' property")
        localFeatureItems: List[FeatureItem] = [
            {"dataset": dataset, "config": config, "feature": {"name": name, "content": content}}
            for (name, content) in infoItem["features"].items()
        ]

        return {"features": localFeatureItems, "rows": rowItems}

    if config is None:
        content = get_configs_response(dataset=dataset).content
        if "configs" not in content:
            error = cast(StatusErrorContent, content)
            if "status_code" in error and error["status_code"] == 404:
                raise Status404Error("configurations could not be found")
            raise Status400Error("configurations could not be found")
        configs_content = cast(ConfigsContent, content)
        configs = [configItem["config"] for configItem in configs_content["configs"]]
    else:
        configs = [config]

    # Note that we raise on the first error
    for config in configs:
        content = get_splits_response(dataset=dataset, config=config).content
        if "splits" not in content:
            error = cast(StatusErrorContent, content)
            if "status_code" in error and error["status_code"] == 404:
                raise Status404Error("splits could not be found")
            raise Status400Error("splits could not be found")
        splits_content = cast(SplitsContent, content)
        splits = [splitItem["split"] for splitItem in splits_content["splits"]]

        for split in splits:
            content = get_rows_response(dataset=dataset, config=config, split=split).content
            if "rows" not in content:
                error = cast(StatusErrorContent, content)
                if "status_code" in error and error["status_code"] == 404:
                    raise Status404Error("rows could not be found")
                raise Status400Error("rows could not be found")
            rows_content = cast(RowsContent, content)
            rowItems += rows_content["rows"]
            for featureItem in rows_content["features"]:
                if featureItem not in featureItems:
                    featureItems.append(featureItem)

    return {"features": featureItems, "rows": rowItems}


@memoize(cache, expire=CACHE_TTL_SECONDS)  # type:ignore
def get_rows_response(*, dataset: str, config: Optional[str] = None, split: Optional[str] = None) -> CachedResponse:
    try:
        response = CachedResponse(get_rows(dataset, config, split))
    except (Status400Error, Status404Error) as err:
        response = CachedResponse(err.as_content(), err.status_code)
    return response


def get_refreshed_rows(dataset: str, config: Optional[str] = None, split: Optional[str] = None) -> RowsContent:
    return cast(RowsContent, get_rows_response(dataset, config, split, _refresh=True)["content"])
