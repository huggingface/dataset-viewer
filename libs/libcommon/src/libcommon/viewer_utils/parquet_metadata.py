from os import makedirs
from pathlib import Path
from urllib import parse

import pyarrow.parquet as pq

from libcommon.constants import DATASET_SEPARATOR
from libcommon.storage import StrPath

PARQUET_METADATA_DIR_MODE = 0o755


def create_parquet_metadata_dir(
    dataset: str, revision: str, config: str, split: str, parquet_metadata_directory: StrPath
) -> tuple[Path, str]:
    parquet_metadata_dir_subpath = f"{parse.quote(dataset)}/{DATASET_SEPARATOR}/{revision}/{DATASET_SEPARATOR}/{parse.quote(config)}/{parse.quote(split)}"
    dir_path = Path(parquet_metadata_directory).resolve() / parquet_metadata_dir_subpath
    makedirs(dir_path, PARQUET_METADATA_DIR_MODE, exist_ok=True)
    return dir_path, parquet_metadata_dir_subpath


def create_parquet_metadata_file(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    parquet_file_metadata: pq.FileMetaData,
    filename: str,
    parquet_metadata_directory: StrPath,
    overwrite: bool = True,
) -> str:
    dir_path, parquet_metadata_dir_subpath = create_parquet_metadata_dir(
        dataset=dataset,
        revision=revision,
        config=config,
        split=split,
        parquet_metadata_directory=parquet_metadata_directory,
    )
    parquet_metadata_file_path = dir_path / filename
    if overwrite or not parquet_metadata_file_path.exists():
        parquet_file_metadata.write_metadata_file(parquet_metadata_file_path)
    parquet_metadata_subpath = f"{parquet_metadata_dir_subpath}/{filename}"
    return parquet_metadata_subpath
