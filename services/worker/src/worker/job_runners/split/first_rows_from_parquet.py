# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from fsspec.implementations.http import HTTPFileSystem
from libcommon.constants import MAX_NUM_ROWS_PER_PAGE
from libcommon.dtos import JobInfo, RowsContent, SplitFirstRowsResponse
from libcommon.exceptions import (
    ParquetResponseEmptyError,
    SplitParquetSchemaMismatchError,
    TooBigContentError,
)
from libcommon.parquet_utils import EmptyParquetMetadataError, Indexer, SchemaMismatchError, TooBigRows
from libcommon.storage import StrPath
from libcommon.storage_client import StorageClient
from libcommon.viewer_utils.rows import create_first_rows_response

from worker.config import AppConfig, FirstRowsConfig
from worker.dtos import CompleteJobResult
from worker.job_runners.split.split_job_runner import SplitJobRunner


def compute_first_rows_response(
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
    logging.info(f"compute 'split-first-rows-from-parquet' for {dataset=} {config=} {split=}")

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
            pa_table = rows_index.query(offset=0, length=rows_max_number)
            return RowsContent(
                rows=pa_table.to_pylist(), all_fetched=rows_index.parquet_index.num_rows_total <= rows_max_number
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


class SplitFirstRowsFromParquetJobRunner(SplitJobRunner):
    first_rows_config: FirstRowsConfig
    indexer: Indexer

    @staticmethod
    def get_job_type() -> str:
        return "split-first-rows-from-parquet"

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        parquet_metadata_directory: StrPath,
        storage_client: StorageClient,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
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
        return CompleteJobResult(
            compute_first_rows_response(
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
