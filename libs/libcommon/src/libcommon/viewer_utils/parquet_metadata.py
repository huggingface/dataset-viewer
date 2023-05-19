from os import makedirs
from pathlib import Path
from typing import Tuple

import pyarrow.parquet as pq

from libcommon.storage import StrPath

DATASET_SEPARATOR = "--"

PARQUET_METADATA_DIR_MODE = 0o755


def create_parquet_metadata_dir(dataset: str, config: str, parquet_metadata_directory: StrPath) -> Tuple[Path, str]:
    dir_path = Path(parquet_metadata_directory).resolve() / dataset / DATASET_SEPARATOR / config
    parquet_metadata_dir_subpath = f"{dataset}/{DATASET_SEPARATOR}/{config}"
    makedirs(dir_path, PARQUET_METADATA_DIR_MODE, exist_ok=True)
    return dir_path, parquet_metadata_dir_subpath


def create_parquet_metadata_file(
    dataset: str,
    config: str,
    parquet_file_metadata: pq.FileMetaData,
    filename: str,
    parquet_metadata_directory: StrPath,
    overwrite: bool = True,
) -> str:
    dir_path, parquet_metadata_dir_subpath = create_parquet_metadata_dir(
        dataset=dataset,
        config=config,
        parquet_metadata_directory=parquet_metadata_directory,
    )
    parquet_metadata_file_path = dir_path / filename
    if overwrite or not parquet_metadata_file_path.exists():
        parquet_file_metadata.write_metadata_file(parquet_metadata_file_path)
    parquet_metadata_subpath = f"{parquet_metadata_dir_subpath}/{filename}"
    return parquet_metadata_subpath
