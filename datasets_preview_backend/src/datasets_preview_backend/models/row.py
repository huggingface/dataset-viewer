import itertools
import logging
from typing import Any, Dict, List, Optional

from datasets import Dataset, DownloadMode, IterableDataset, load_dataset

from datasets_preview_backend.constants import DEFAULT_ROWS_MAX_NUMBER
from datasets_preview_backend.utils import retry

logger = logging.getLogger(__name__)


Row = Dict[str, Any]


@retry(logger=logger)
def get_rows(
    dataset_name: str,
    config_name: str,
    split_name: str,
    hf_token: Optional[str] = None,
    streaming: bool = True,
    rows_max_number: Optional[int] = None,
) -> List[Row]:
    if rows_max_number is None:
        rows_max_number = DEFAULT_ROWS_MAX_NUMBER
    dataset = load_dataset(
        dataset_name,
        name=config_name,
        split=split_name,
        streaming=streaming,
        download_mode=DownloadMode.FORCE_REDOWNLOAD,
        use_auth_token=hf_token,
    )
    if streaming:
        if not isinstance(dataset, IterableDataset):
            raise TypeError("load_dataset should return an IterableDataset")
    elif not isinstance(dataset, Dataset):
        raise TypeError("load_dataset should return a Dataset")
    rows_plus_one = list(itertools.islice(dataset, rows_max_number + 1))
    # ^^ to be able to detect if a split has exactly ROWS_MAX_NUMBER rows
    if len(rows_plus_one) <= rows_max_number:
        logger.debug(f"all the rows in the split have been fetched ({len(rows_plus_one)})")
    else:
        logger.debug(f"the rows in the split have been truncated ({rows_max_number} rows)")
    return rows_plus_one[:rows_max_number]
