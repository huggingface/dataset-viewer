import os
from typing import Tuple

from appdirs import user_cache_dir  # type:ignore

from datasets_preview_backend.config import ASSETS_DIRECTORY

DATASET_SEPARATOR = "___"
ASSET_DIR_MODE = 755

# set it to the default cache location on the machine, if ASSETS_DIRECTORY is null
assets_directory = user_cache_dir("datasets_preview_backend_assets") if ASSETS_DIRECTORY is None else ASSETS_DIRECTORY
os.makedirs(assets_directory, exist_ok=True)


def create_asset_dir(dataset: str, config: str, split: str, row_idx: int, column: str) -> Tuple[str, str]:
    dir_path = os.path.join(assets_directory, dataset, DATASET_SEPARATOR, config, split, str(row_idx), column)
    url_dir_path = f"{dataset}/{DATASET_SEPARATOR}/{config}/{split}/{row_idx}/{column}"
    os.makedirs(dir_path, ASSET_DIR_MODE, exist_ok=True)
    return dir_path, url_dir_path


def create_asset_file(
    dataset: str, config: str, split: str, row_idx: int, column: str, filename: str, data: bytes
) -> str:
    dir_path, url_dir_path = create_asset_dir(dataset, config, split, row_idx, column)
    file_path = os.path.join(dir_path, filename)
    with open(file_path, "wb") as f:
        f.write(data)
    return f"assets/{url_dir_path}/{filename}"


# TODO: add a function to flush all the assets of a dataset
