import logging
import os
from typing import Optional

from appdirs import user_cache_dir  # type:ignore

logger = logging.getLogger(__name__)

DATASET_SEPARATOR = "--"
ASSET_DIR_MODE = 0o755


def init_assets_dir(assets_directory: Optional[str] = None) -> str:
    # set it to the default cache location on the machine, if ASSETS_DIRECTORY is null
    if assets_directory is None:
        assets_directory = user_cache_dir("datasets_preview_backend_assets")
    os.makedirs(assets_directory, exist_ok=True)
    return assets_directory


def show_assets_dir(assets_directory: Optional[str] = None) -> None:
    logger.info(f"Assets directory: {init_assets_dir(assets_directory)}")
