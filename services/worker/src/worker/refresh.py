import logging
from typing import List, Optional

from libcache.cache import (
    upsert_dataset,
    upsert_dataset_error,
    upsert_json_split,
    upsert_split_error,
)
from libutils.exceptions import Status500Error, StatusError
from libutils.types import SplitFullName

from worker.models.dataset import get_dataset_split_full_names
from worker.models.split import get_json_split

logger = logging.getLogger(__name__)


def refresh_dataset_split_full_names(dataset_name: str, hf_token: Optional[str] = None) -> List[SplitFullName]:
    try:
        split_full_names = get_dataset_split_full_names(dataset_name, hf_token)
        upsert_dataset(dataset_name, split_full_names)
        logger.debug(f"dataset '{dataset_name}' is valid, cache updated")
        return split_full_names
    except StatusError as err:
        upsert_dataset_error(dataset_name, err)
        logger.debug(f"dataset '{dataset_name}' had error, cache updated")
        raise
    except Exception as err:
        upsert_dataset_error(dataset_name, Status500Error(str(err)))
        logger.debug(f"dataset '{dataset_name}' had error, cache updated")
        raise


def refresh_split(
    dataset_name: str,
    config_name: str,
    split_name: str,
    hf_token: Optional[str] = None,
    max_size_fallback: Optional[int] = None,
    rows_max_bytes: Optional[int] = None,
    rows_max_number: Optional[int] = None,
    rows_min_number: Optional[int] = None,
):
    try:
        json_split = get_json_split(
            dataset_name,
            config_name,
            split_name,
            hf_token=hf_token,
            max_size_fallback=max_size_fallback,
            rows_max_bytes=rows_max_bytes,
            rows_max_number=rows_max_number,
            rows_min_number=rows_min_number,
        )
        upsert_json_split(dataset_name, config_name, split_name, json_split)
        logger.debug(
            f"split '{split_name}' from dataset '{dataset_name}' in config '{config_name}' is valid, cache updated"
        )
    except StatusError as err:
        upsert_split_error(dataset_name, config_name, split_name, err)
        logger.debug(
            f"split '{split_name}' from dataset '{dataset_name}' in config '{config_name}' had error, cache updated"
        )
        raise
    except Exception as err:
        upsert_split_error(dataset_name, config_name, split_name, Status500Error(str(err)))
        logger.debug(
            f"split '{split_name}' from dataset '{dataset_name}' in config '{config_name}' had error, cache updated"
        )
        raise
