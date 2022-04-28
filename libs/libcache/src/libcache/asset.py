import logging
import os

from appdirs import user_cache_dir  # type:ignore
from libcache.config import ASSETS_DIRECTORY

logger = logging.getLogger(__name__)

DATASET_SEPARATOR = "--"
ASSET_DIR_MODE = 0o755


def get_assets_dir() -> str:
    # set it to the default cache location on the machine, if ASSETS_DIRECTORY is null
    assets_directory = (
        user_cache_dir("datasets_preview_backend_assets") if ASSETS_DIRECTORY is None else ASSETS_DIRECTORY
    )
    os.makedirs(assets_directory, exist_ok=True)
    return assets_directory


def show_assets_dir() -> None:
    logger.info(f"Assets directory: {get_assets_dir()}")
