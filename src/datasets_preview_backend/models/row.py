import logging
import re
from typing import Any, Dict, List

from datasets import IterableDataset, load_dataset

from datasets_preview_backend.config import EXTRACT_ROWS_LIMIT
from datasets_preview_backend.constants import FORCE_REDOWNLOAD

from datasets_preview_backend.exceptions import Status400Error, Status404Error

logger = logging.getLogger(__name__)


Row = Dict[str, Any]


def get_rows(dataset_name: str, config_name: str, split_name: str) -> List[Row]:
    num_rows = EXTRACT_ROWS_LIMIT
    try:
        iterable_dataset = load_dataset(
            dataset_name,
            name=config_name,
            split=split_name,
            streaming=True,
            download_mode=FORCE_REDOWNLOAD,  # type: ignore
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
            str(err).startswith(f"BuilderConfig {config_name} not found.")
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
            f"could not read all the required rows ({len(rows)} / {num_rows}) from dataset {dataset_name} -"
            f" {config_name} - {split_name}"
        )

    return rows
