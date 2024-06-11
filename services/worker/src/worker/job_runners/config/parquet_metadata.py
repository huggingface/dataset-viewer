# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import functools
import logging
from typing import Optional

import aiohttp
from huggingface_hub import HfFileSystemFile
from libcommon.dtos import JobInfo, SplitHubFile
from libcommon.exceptions import (
    FileSystemError,
    ParquetResponseEmptyError,
    PreviousStepFormatError,
)
from libcommon.parquet_utils import extract_split_name_from_parquet_url
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.storage import StrPath
from libcommon.utils import retry
from libcommon.viewer_utils.parquet_metadata import create_parquet_metadata_file
from pyarrow.parquet import ParquetFile
from tqdm.contrib.concurrent import thread_map

from worker.config import AppConfig
from worker.dtos import (
    CompleteJobResult,
    ConfigParquetMetadataResponse,
    ParquetFileMetadataItem,
)
from worker.job_runners.config.config_job_runner import ConfigJobRunner
from worker.utils import open_file

SLEEPS = [0.2, 1, 1, 10, 10, 10]


# TODO: on which error?
@retry(on=[aiohttp.ServerConnectionError], sleeps=SLEEPS)
def retry_open_file(
    file_url: str, hf_endpoint: str, hf_token: Optional[str], revision: Optional[str] = None
) -> HfFileSystemFile:
    return open_file(file_url=file_url, hf_endpoint=hf_endpoint, hf_token=hf_token, revision=revision)


def create_parquet_metadata_file_from_remote_parquet(
    parquet_file_item: SplitHubFile, hf_endpoint: str, hf_token: Optional[str], parquet_metadata_directory: StrPath
) -> ParquetFileMetadataItem:
    split_directory = extract_split_name_from_parquet_url(parquet_file_item["url"])
    hfh_file_url = f"datasets/{parquet_file_item['config']}/{split_directory}/{parquet_file_item['filename']}"
    try:
        # TODO: make revision config's parameter
        f = retry_open_file(
            file_url=hfh_file_url, hf_endpoint=hf_endpoint, hf_token=hf_token, revision="refs/convert/parquet"
        )
        parquet_file_metadata = ParquetFile(f).metadata
    except Exception as e:
        raise FileSystemError(f"Could not read the parquet files: {e}") from e
    split = parquet_file_item["url"].split("/")[-2]
    # ^ https://github.com/huggingface/dataset-viewer/issues/2768
    # to support more than 10k parquet files, in which case, instead of "train" for example,
    # the subdirectories are "train-part0", "train-part1", "train-part2", etc.
    parquet_metadata_subpath = create_parquet_metadata_file(
        dataset=parquet_file_item["dataset"],
        config=parquet_file_item["config"],
        split=split,
        parquet_file_metadata=parquet_file_metadata,
        filename=parquet_file_item["filename"],
        parquet_metadata_directory=parquet_metadata_directory,
    )
    f.close()
    return ParquetFileMetadataItem(
        dataset=parquet_file_item["dataset"],
        config=parquet_file_item["config"],
        split=parquet_file_item["split"],
        url=parquet_file_item["url"],
        filename=parquet_file_item["filename"],
        size=parquet_file_item["size"],
        num_rows=parquet_file_metadata.num_rows,
        parquet_metadata_subpath=parquet_metadata_subpath,
    )


def compute_parquet_metadata_response(
    dataset: str, config: str, hf_endpoint: str, hf_token: Optional[str], parquet_metadata_directory: StrPath
) -> ConfigParquetMetadataResponse:
    """
    Get the response of 'config-parquet-metadata' for one specific dataset and config on huggingface.co.
    Store the config's parquet metadata on the disk and return the list of local metadata files.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            A configuration name.
        hf_endpoint (`str`):
            The Hub endpoint (for example: "https://huggingface.co")
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
        parquet_metadata_directory (`str` or `pathlib.Path`):
            The directory where the parquet metadata files are stored.

    Raises:
        [~`libcommon.simple_cache.CachedArtifactError`]:
            If the previous step gave an error.
        [~`libcommon.exceptions.PreviousStepFormatError`]:
            If the content of the previous step has not the expected format
        [~`libcommon.exceptions.ParquetResponseEmptyError`]:
            If the previous step provided an empty list of parquet files.
        [~`libcommon.exceptions.FileSystemError`]:
            If the HfFileSystem couldn't access the parquet files.

    Returns:
        `ConfigParquetMetadataResponse`: An object with the list of parquet metadata files.
    """
    logging.info(f"compute 'config-parquet-metadata' for {dataset=} {config=}")

    config_parquet_response = get_previous_step_or_raise(kind="config-parquet", dataset=dataset, config=config)
    try:
        parquet_files_content = config_parquet_response["content"]["parquet_files"]
        parquet_file_items: list[SplitHubFile] = [
            parquet_file_item for parquet_file_item in parquet_files_content if parquet_file_item["config"] == config
        ]
        if not parquet_file_items:
            raise ParquetResponseEmptyError("No parquet files found.")
        content = config_parquet_response["content"]
        if "features" in content and isinstance(content["features"], dict):
            features = content["features"]  # config-parquet version<6 didn't have features
        else:
            # (July 23) we can remove this later and raise an error instead (can be None for backward compatibility)
            features = None
        partial = config_parquet_response["content"]["partial"]
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.") from e

    desc = f"{dataset}/{config}"
    parquet_files_metadata: list[ParquetFileMetadataItem] = thread_map(
        functools.partial(
            create_parquet_metadata_file_from_remote_parquet,
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            parquet_metadata_directory=parquet_metadata_directory,
        ),
        parquet_file_items,
        desc=desc,
        unit="pq",
        disable=True,
    )
    return ConfigParquetMetadataResponse(
        parquet_files_metadata=parquet_files_metadata, features=features, partial=partial
    )


class ConfigParquetMetadataJobRunner(ConfigJobRunner):
    parquet_metadata_directory: StrPath

    @staticmethod
    def get_job_type() -> str:
        return "config-parquet-metadata"

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        parquet_metadata_directory: StrPath,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
        )
        self.parquet_metadata_directory = parquet_metadata_directory

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_parquet_metadata_response(
                dataset=self.dataset,
                config=self.config,
                hf_endpoint=self.app_config.common.hf_endpoint,
                hf_token=self.app_config.common.hf_token,
                parquet_metadata_directory=self.parquet_metadata_directory,
            )
        )
