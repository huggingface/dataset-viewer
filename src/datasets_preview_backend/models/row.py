import functools
import logging
import time
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, DownloadMode, IterableDataset, load_dataset

from datasets_preview_backend.config import ROWS_MAX_NUMBER

logger = logging.getLogger(__name__)


Row = Dict[str, Any]


def retry(func):
    """retries with an increasing sleep before every attempt"""
    SLEEPS = [1, 7, 70, 7 * 60, 70 * 60]
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


def take_rows(dataset: Union[Dataset, IterableDataset]) -> List[Row]:
    iterator = iter(dataset)

    i = 0
    rows = []
    while True:
        try:
            row = next(iterator)
        except StopIteration:
            logger.debug(f"all the rows have been fetched ({i})")
            break
        if i >= ROWS_MAX_NUMBER:
            logger.debug("reached max number of rows ({ROWS_MAX_NUMBER}): truncate")
            break
        rows.append(row)
        i += 1

    return rows


@retry
def get_rows(
    dataset_name: str, config_name: str, split_name: str, hf_token: Optional[str] = None, streaming: bool = True
) -> List[Row]:
    dataset = load_dataset(
        dataset_name,
        name=config_name,
        split=split_name,
        streaming=True,
        download_mode=DownloadMode.FORCE_REDOWNLOAD,
        use_auth_token=hf_token,
    )
    if streaming:
        if not isinstance(dataset, IterableDataset):
            raise TypeError("load_dataset should return an IterableDataset")
    elif not isinstance(dataset, Dataset):
        raise TypeError("load_dataset should return a Dataset")
    return take_rows(dataset)
