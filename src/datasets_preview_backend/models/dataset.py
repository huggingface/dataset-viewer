import logging
from typing import List, Optional, TypedDict

from datasets import get_dataset_config_names, get_dataset_split_names

from datasets_preview_backend.constants import FORCE_REDOWNLOAD
from datasets_preview_backend.exceptions import Status400Error

logger = logging.getLogger(__name__)


class SplitFullName(TypedDict):
    dataset_name: str
    config_name: str
    split_name: str


def get_dataset_split_full_names(dataset_name: str, hf_token: Optional[str] = None) -> List[SplitFullName]:
    logger.info(f"get dataset '{dataset_name}' split full names")
    try:
        return [
            {"dataset_name": dataset_name, "config_name": config_name, "split_name": split_name}
            for config_name in get_dataset_config_names(
                dataset_name, download_mode=FORCE_REDOWNLOAD, use_auth_token=hf_token  # type: ignore
            )
            for split_name in get_dataset_split_names(
                dataset_name, config_name, use_auth_token=hf_token  # type: ignore
            )
        ]
    except Exception as err:
        raise Status400Error("Cannot get the split names for the dataset.", err)
