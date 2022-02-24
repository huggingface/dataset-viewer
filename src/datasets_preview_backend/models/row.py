import itertools
import logging
from typing import Any, Dict, List, Optional

from datasets import Dataset, DownloadMode, IterableDataset, load_dataset

from datasets_preview_backend.config import ROWS_MAX_NUMBER
from datasets_preview_backend.utils import retry

logger = logging.getLogger(__name__)


Row = Dict[str, Any]


@retry(logger=logger)
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
    rows_plus_one = list(itertools.islice(dataset, ROWS_MAX_NUMBER + 1))
    # ^^ to be able to detect if a split has exactly DEFAULT_ROWS_MAX_NUMBER rows
    if len(rows_plus_one) <= ROWS_MAX_NUMBER:
        logger.debug(f"all the rows in the split have been fetched ({len(rows_plus_one)})")
    else:
        logger.debug(f"the rows in the split have been truncated ({ROWS_MAX_NUMBER} rows)")
    return rows_plus_one[:ROWS_MAX_NUMBER]
    # ^^ note that DEFAULT_ROWS_MAX_BYTES is not enforced here, but in typed_row.py
    # after the type of the fields is known (ie: the row can be converted to JSON)
