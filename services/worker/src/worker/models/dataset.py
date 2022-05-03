import logging
from typing import List, Optional

from datasets import DownloadMode, get_dataset_config_names, get_dataset_split_names
from datasets.inspect import SplitsNotFoundError
from libutils.exceptions import Status400Error
from libutils.types import SplitFullName

from worker.models._guard import guard_blocked_datasets

logger = logging.getLogger(__name__)


def get_dataset_split_full_names(dataset_name: str, hf_token: Optional[str] = None) -> List[SplitFullName]:
    logger.info(f"get dataset '{dataset_name}' split full names")

    try:
        guard_blocked_datasets(dataset_name)
        return [
            {"dataset_name": dataset_name, "config_name": config_name, "split_name": split_name}
            for config_name in get_dataset_config_names(
                dataset_name, download_mode=DownloadMode.FORCE_REDOWNLOAD, use_auth_token=hf_token
            )
            for split_name in get_dataset_split_names(dataset_name, config_name, use_auth_token=hf_token)
        ]
    except SplitsNotFoundError as err:
        # we bypass the SplitsNotFoundError, as we're interested in the cause
        raise Status400Error("Cannot get the split names for the dataset.", err.__cause__) from err
    except Exception as err:
        raise Status400Error("Cannot get the split names for the dataset.", err) from err
