import logging
import re
from typing import Optional, Union, cast

from datasets import IterableDataset, load_dataset

from datasets_preview_backend.cache import memoize  # type: ignore
from datasets_preview_backend.config import CACHE_TTL_SECONDS, EXTRACT_ROWS_LIMIT, cache
from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.responses import CachedResponse
from datasets_preview_backend.types import RowsDict

logger = logging.getLogger(__name__)


def extract_rows(dataset: str, config: Union[str, None], split: str, token: Optional[str] = None) -> RowsDict:
    if not isinstance(dataset, str) and dataset is not None:
        raise TypeError("dataset argument should be a string")
    if dataset is None:
        raise Status400Error("'dataset' is a required query parameter.")
    config = DEFAULT_CONFIG_NAME if config is None else config
    if not isinstance(config, str) and config is not None:
        raise TypeError("config argument should be a string")
    if not isinstance(split, str) and split is not None:
        raise TypeError("split argument should be a string")
    if split is None:
        raise Status400Error("'split' is a required query parameter.")
    num_rows = EXTRACT_ROWS_LIMIT

    try:
        iterable_dataset = load_dataset(dataset, name=config, split=split, streaming=True, use_auth_token=token)
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
            f"could not read all the required rows ({len(rows)} / {num_rows}) from dataset {dataset} - {config} -"
            f" {split}"
        )

    return {
        "dataset": dataset,
        "config": config,
        "split": split,
        "rows": rows,
    }


@memoize(cache, expire=CACHE_TTL_SECONDS)  # type:ignore
def get_rows_response(
    *, dataset: str, config: Union[str, None], split: str, token: Optional[str] = None
) -> CachedResponse:
    try:
        response = CachedResponse(extract_rows(dataset, config, split, token))
    except (Status400Error, Status404Error) as err:
        response = CachedResponse(err.as_dict(), err.status_code)
    return response


def get_refreshed_rows(dataset: str, config: Union[str, None], split: str, token: Optional[str] = None) -> RowsDict:
    return cast(RowsDict, get_rows_response(dataset, config, split, token, _refresh=True)["content"])
