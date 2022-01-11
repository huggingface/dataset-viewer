import logging
from typing import Any, Dict, List, Optional

from datasets import Dataset, IterableDataset, load_dataset

from datasets_preview_backend.config import EXTRACT_ROWS_LIMIT
from datasets_preview_backend.constants import FORCE_REDOWNLOAD
from datasets_preview_backend.exceptions import Status400Error

logger = logging.getLogger(__name__)


Row = Dict[str, Any]


def get_rows_without_streaming(
    dataset_name: str,
    config_name: str,
    split_name: str,
    hf_token: Optional[str] = None,
) -> List[Row]:
    num_rows = EXTRACT_ROWS_LIMIT
    try:
        dataset = load_dataset(
            dataset_name,
            name=config_name,
            split=split_name,
            use_auth_token=hf_token,
        )
        if not isinstance(dataset, Dataset):
            raise TypeError("load_dataset should return a Dataset")
        d = dataset[:num_rows]
        size = len(next(iter(d.values())))
        rows = [{col: d[col][i] for col in d} for i in range(size)]
    except Exception as err:
        raise Status400Error("Cannot get the first rows for the split.", err)
    if len(rows) != num_rows:
        logger.info(
            f"could not read all the required rows ({len(rows)} / {num_rows}) from dataset {dataset_name} -"
            f" {config_name} - {split_name}"
        )
    return rows


def get_rows(dataset_name: str, config_name: str, split_name: str, hf_token: Optional[str] = None) -> List[Row]:
    num_rows = EXTRACT_ROWS_LIMIT
    try:
        iterable_dataset = load_dataset(
            dataset_name,
            name=config_name,
            split=split_name,
            streaming=True,
            download_mode=FORCE_REDOWNLOAD,  # type: ignore
            use_auth_token=hf_token,
        )
        if not isinstance(iterable_dataset, IterableDataset):
            raise TypeError("load_dataset should return an IterableDataset")
        rows = list(iterable_dataset.take(num_rows))
    except Exception as err:
        raise Status400Error("Cannot get the first rows for the split.", err)
    if len(rows) != num_rows:
        logger.info(
            f"could not read all the required rows ({len(rows)} / {num_rows}) from dataset {dataset_name} -"
            f" {config_name} - {split_name}"
        )
    return rows
