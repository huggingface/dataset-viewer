import functools
import logging
import time
from typing import Any, Dict, List, Optional

from datasets import Dataset, IterableDataset, load_dataset

from datasets_preview_backend.config import EXTRACT_ROWS_LIMIT
from datasets_preview_backend.constants import FORCE_REDOWNLOAD
from datasets_preview_backend.exceptions import Status400Error

logger = logging.getLogger(__name__)


Row = Dict[str, Any]


def retry(func):
    """retries with an increasing sleep before every attempt"""
    SLEEPS = [7, 70, 7 * 60, 70 * 60]
    MAX_ATTEMPTS = len(SLEEPS)

    @functools.wraps(func)
    def decorator(*args, **kwargs):
        attempt = 0
        while attempt < MAX_ATTEMPTS:
            try:
                """always sleep before calling the function. It will prevent rate limiting in the first place"""
                duration = SLEEPS[attempt]
                logger.info(f"Sleep during {duration} seconds to preventively mitigate rate limiting.")
                time.sleep(duration)
                return func(*args, **kwargs)
            except ConnectionError:
                logger.info("Got a ConnectionError, possibly due to rate limiting. Let's retry.")
                attempt += 1
        raise Exception(f"Give up after {attempt} attempts with ConnectionError")

    return decorator


@retry
def extract_rows(
    dataset_name: str, config_name: str, split_name: str, num_rows: int, hf_token: Optional[str] = None
) -> List[Row]:
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
    return list(iterable_dataset.take(num_rows))


def get_rows(dataset_name: str, config_name: str, split_name: str, hf_token: Optional[str] = None) -> List[Row]:
    num_rows = EXTRACT_ROWS_LIMIT
    try:
        rows = extract_rows(dataset_name, config_name, split_name, num_rows, hf_token)
    except Exception as err:
        raise Status400Error("Cannot get the first rows for the split.", err) from err
    if len(rows) != num_rows:
        logger.info(
            f"could not read all the required rows ({len(rows)} / {num_rows}) from dataset {dataset_name} -"
            f" {config_name} - {split_name}"
        )
    return rows


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
        raise Status400Error("Cannot get the first rows for the split.", err) from err
    if len(rows) != num_rows:
        logger.info(
            f"could not read all the required rows ({len(rows)} / {num_rows}) from dataset {dataset_name} -"
            f" {config_name} - {split_name}"
        )
    return rows
