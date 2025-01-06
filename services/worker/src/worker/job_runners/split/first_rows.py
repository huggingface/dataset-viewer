# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.


import logging
from pathlib import Path
from typing import Optional

from datasets import IterableDataset, get_dataset_config_info, load_dataset
from fsspec.implementations.http import HTTPFileSystem
from libcommon.constants import MAX_NUM_ROWS_PER_PAGE
from libcommon.dtos import JobInfo, RowsContent, SplitFirstRowsResponse
from libcommon.exceptions import (
    DatasetWithScriptNotSupportedError,
    FeaturesError,
    InfoError,
    ParquetResponseEmptyError,
    SplitParquetSchemaMismatchError,
    TooBigContentError,
)
from libcommon.parquet_utils import EmptyParquetMetadataError, Indexer, SchemaMismatchError, TooBigRows
from libcommon.simple_cache import CachedArtifactError, CachedArtifactNotFoundError
from libcommon.storage import StrPath
from libcommon.storage_client import StorageClient
from libcommon.viewer_utils.rows import create_first_rows_response

from worker.config import AppConfig, FirstRowsConfig
from worker.dtos import CompleteJobResult
from worker.job_runners.split.split_job_runner import SplitJobRunnerWithDatasetsCache
from worker.utils import get_rows_or_raise, raise_if_long_column_name


def compute_first_rows_from_parquet_response(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    storage_client: StorageClient,
    min_cell_bytes: int,
    rows_max_bytes: int,
    rows_max_number: int,
    rows_min_number: int,
    columns_max_number: int,
    indexer: Indexer,
) -> SplitFirstRowsResponse:
    """
    Compute the response of 'split-first-rows' for one specific split of a dataset from the parquet files.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.
        revision (`str`):
            The git revision of the dataset.
        config (`str`):
            A configuration name.
        split (`str`):
            A split name.
        storage_client (`StorageClient`):
            A storage client to save the assets (images, audio, etc.).
        min_cell_bytes (`int`):
            The minimum number of bytes for a cell, when truncation applies.
        rows_max_bytes (`int`):
            The maximum number of bytes of the response (else, the response is truncated).
        rows_max_number (`int`):
            The maximum number of rows of the response.
        rows_min_number (`int`):
            The minimum number of rows of the response.
        columns_max_number (`int`):
            The maximum number of columns supported.
        indexer (`Indexer`):
            An indexer to get the rows index.

    Raises:
        [~`libcommon.exceptions.ParquetResponseEmptyError`]:
          If the parquet files are empty.
        [~`libcommon.exceptions.SplitParquetSchemaMismatchError`]:
          If the parquet files have different schemas.
        [~`libcommon.exceptions.TooBigContentError`]:
          If the first rows content exceeds the maximum supported size of bytes.

    Returns:
        `SplitFirstRowsResponse`: The list of first rows of the split.
    """

    logging.info(f"compute 'split-first-rows' from parquet for {dataset=} {config=} {split=}")

    try:
        rows_index = indexer.get_rows_index(
            dataset=dataset,
            config=config,
            split=split,
        )
    except EmptyParquetMetadataError:
        raise ParquetResponseEmptyError("No parquet files found.")

    features = rows_index.parquet_index.features

    # get the rows
    def get_rows_content(rows_max_number: int) -> RowsContent:
        try:
            truncated_columns: list[str] = []
            if dataset == "Major-TOM/Core-S2L2A":
                pa_table, truncated_columns = rows_index.query_truncated_binary(offset=0, length=rows_max_number)
            else:
                pa_table = rows_index.query(offset=0, length=rows_max_number)
            return RowsContent(
                rows=pa_table.to_pylist(),
                all_fetched=rows_index.parquet_index.num_rows_total <= rows_max_number,
                truncated_columns=truncated_columns,
            )
        except TooBigRows as err:
            raise TooBigContentError(str(err))
        except SchemaMismatchError as err:
            raise SplitParquetSchemaMismatchError(
                "Split parquet files being processed have different schemas. Ensure all files have identical column names.",
                cause=err,
            )

    return create_first_rows_response(
        dataset=dataset,
        revision=revision,
        config=config,
        split=split,
        storage_client=storage_client,
        features=features,
        get_rows_content=get_rows_content,
        min_cell_bytes=min_cell_bytes,
        rows_max_bytes=rows_max_bytes,
        rows_max_number=rows_max_number,
        rows_min_number=rows_min_number,
        columns_max_number=columns_max_number,
    )


def compute_first_rows_from_streaming_response(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    storage_client: StorageClient,
    hf_token: Optional[str],
    min_cell_bytes: int,
    rows_max_bytes: int,
    rows_max_number: int,
    rows_min_number: int,
    columns_max_number: int,
    max_size_fallback: Optional[int] = None,
) -> SplitFirstRowsResponse:
    """
    Get the response of 'split-first-rows' using streaming for one specific split of a dataset from huggingface.co.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.
        revision (`str`):
            The git revision of the dataset.
        config (`str`):
            A configuration name.
        split (`str`):
            A split name.
        storage_client (`StorageClient`):
            A storage client to save the assets (images, audio, etc.).
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
        min_cell_bytes (`int`):
            The minimum number of bytes for a cell, when truncation applies.
        rows_max_bytes (`int`):
            The maximum number of bytes of the response (else, the response is truncated).
        rows_max_number (`int`):
            The maximum number of rows of the response.
        rows_min_number (`int`):
            The minimum number of rows of the response.
        columns_max_number (`int`):
            The maximum number of columns supported.
        max_size_fallback (`int`, *optional*): **DEPRECATED**
            The maximum number of bytes of the split to fallback to normal mode if the streaming mode fails.
            This argument is now hard-coded to 100MB, and will be removed in a future version.

    Raises:
        [~`libcommon.exceptions.SplitNotFoundError`]:
          If the split does not exist in the dataset.
        [~`libcommon.exceptions.InfoError`]:
          If the config info could not be obtained using the datasets library.
        [~`libcommon.exceptions.FeaturesError`]:
          If the split features could not be obtained using the datasets library.
        [~`libcommon.exceptions.RowsPostProcessingError`]:
          If the post-processing of the split rows failed, e.g. while saving the images or audio files to the assets.
        [~`libcommon.exceptions.TooManyColumnsError`]:
          If the number of columns (features) exceeds the maximum supported number of columns.
        [~`libcommon.exceptions.TooBigContentError`]:
          If the first rows content exceeds the maximum supported size of bytes.
        [~`libcommon.simple_cache.CachedArtifactError`]:
          If the previous step gave an error.
        [~`libcommon.exceptions.PreviousStepFormatError`]:
          If the content of the previous step has not the expected format
        [~`libcommon.exceptions.StreamingRowsError`]:
          If the split rows could not be obtained using the datasets library in streaming mode.
        [~`libcommon.exceptions.NormalRowsError`]:
          If the split rows could not be obtained using the datasets library in normal mode.
        [~`libcommon.exceptions.DatasetWithScriptNotSupportedError`]:
            If the dataset has a dataset script.
        [~`libcommon.exceptions.TooLongColumnNameError`]:
            If one of the columns' name is too long (> 500 characters)

    Returns:
        `SplitFirstRowsResponse`: The list of first rows of the split.
    """
    logging.info(f"compute 'split-first-rows' from streaming for {dataset=} {config=} {split=}")
    # get the features
    try:
        info = get_dataset_config_info(path=dataset, config_name=config, token=hf_token)
    except Exception as err:
        if isinstance(err, ValueError) and "trust_remote_code" in str(err):
            raise DatasetWithScriptNotSupportedError from err
        raise InfoError(
            f"The info cannot be fetched for the config '{config}' of the dataset.",
            cause=err,
        ) from err
    if not info.features:
        try:
            # https://github.com/huggingface/datasets/blob/f5826eff9b06ab10dba1adfa52543341ef1e6009/src/datasets/iterable_dataset.py#L1255
            iterable_dataset = load_dataset(
                path=dataset,
                name=config,
                split=split,
                streaming=True,
                token=hf_token,
            )
            if not isinstance(iterable_dataset, IterableDataset):
                raise TypeError("load_dataset should return an IterableDataset.")
            iterable_dataset = iterable_dataset._resolve_features()
            if not isinstance(iterable_dataset, IterableDataset):
                raise TypeError("load_dataset should return an IterableDataset.")
            features = iterable_dataset.features
        except Exception as err:
            if isinstance(err, ValueError) and "trust_remote_code" in str(err):
                raise DatasetWithScriptNotSupportedError from err
            raise FeaturesError(
                (
                    f"Cannot extract the features (columns) for the split '{split}' of the config '{config}' of the"
                    " dataset."
                ),
                cause=err,
            ) from err
    else:
        features = info.features
    raise_if_long_column_name(features)

    def get_rows_content(rows_max_number: int) -> RowsContent:
        return get_rows_or_raise(
            dataset=dataset,
            config=config,
            split=split,
            info=info,
            max_size_fallback=max_size_fallback,
            rows_max_number=rows_max_number,
            token=hf_token,
        )

    return create_first_rows_response(
        dataset=dataset,
        revision=revision,
        config=config,
        split=split,
        storage_client=storage_client,
        features=features,
        get_rows_content=get_rows_content,
        min_cell_bytes=min_cell_bytes,
        rows_max_bytes=rows_max_bytes,
        rows_max_number=rows_max_number,
        rows_min_number=rows_min_number,
        columns_max_number=columns_max_number,
    )


class SplitFirstRowsJobRunner(SplitJobRunnerWithDatasetsCache):
    first_rows_config: FirstRowsConfig
    indexer: Indexer

    @staticmethod
    def get_job_type() -> str:
        return "split-first-rows"

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        hf_datasets_cache: Path,
        parquet_metadata_directory: StrPath,
        storage_client: StorageClient,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            hf_datasets_cache=hf_datasets_cache,
        )
        self.first_rows_config = app_config.first_rows
        self.parquet_metadata_directory = parquet_metadata_directory
        self.indexer = Indexer(
            hf_token=self.app_config.common.hf_token,
            parquet_metadata_directory=parquet_metadata_directory,
            httpfs=HTTPFileSystem(headers={"authorization": f"Bearer {self.app_config.common.hf_token}"}),
            unsupported_features=[],
            all_columns_supported_datasets_allow_list="all",
            max_arrow_data_in_memory=app_config.rows_index.max_arrow_data_in_memory,
        )
        self.storage_client = storage_client

    def compute(self) -> CompleteJobResult:
        try:
            return CompleteJobResult(
                compute_first_rows_from_parquet_response(
                    dataset=self.dataset,
                    revision=self.dataset_git_revision,
                    config=self.config,
                    split=self.split,
                    storage_client=self.storage_client,
                    min_cell_bytes=self.first_rows_config.min_cell_bytes,
                    rows_max_bytes=self.first_rows_config.max_bytes,
                    rows_min_number=self.first_rows_config.min_number,
                    rows_max_number=MAX_NUM_ROWS_PER_PAGE,
                    columns_max_number=self.first_rows_config.columns_max_number,
                    indexer=self.indexer,
                )
            )
        except (
            ParquetResponseEmptyError,
            SplitParquetSchemaMismatchError,
            CachedArtifactNotFoundError,
            CachedArtifactError,
        ):
            logging.info(
                f"Cannot compute 'split-first-rows' from parquet for {self.dataset=} {self.config=}. "
                f"Trying to compute it using streaming."
            )
            pass
        return CompleteJobResult(
            compute_first_rows_from_streaming_response(
                dataset=self.dataset,
                revision=self.dataset_git_revision,
                config=self.config,
                split=self.split,
                storage_client=self.storage_client,
                hf_token=self.app_config.common.hf_token,
                min_cell_bytes=self.first_rows_config.min_cell_bytes,
                rows_max_bytes=self.first_rows_config.max_bytes,
                rows_min_number=self.first_rows_config.min_number,
                rows_max_number=MAX_NUM_ROWS_PER_PAGE,
                columns_max_number=self.first_rows_config.columns_max_number,
            )
        )
