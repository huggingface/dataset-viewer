import logging
from typing import Any, Dict, Optional

from datasets import DatasetInfo, DownloadMode, get_dataset_config_info

from datasets_preview_backend.exceptions import Status400Error

logger = logging.getLogger(__name__)

Info = Dict[str, Any]


def get_info(dataset_name: str, config_name: str, hf_token: Optional[str] = None) -> DatasetInfo:
    logger.info(f"get info metadata for config '{config_name}' of dataset '{dataset_name}'")
    try:
        info = get_dataset_config_info(
            dataset_name,
            config_name=config_name,
            download_mode=DownloadMode.FORCE_REDOWNLOAD,
            use_auth_token=hf_token,
        )
    except Exception as err:
        raise Status400Error("Cannot get the metadata info for the config.", err) from err
    return info
