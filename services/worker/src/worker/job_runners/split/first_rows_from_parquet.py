# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from datasets import Audio, Features, Image
from fsspec.implementations.http import HTTPFileSystem
from libcommon.exceptions import (
    ParquetResponseEmptyError,
    RowsPostProcessingError,
    SplitParquetSchemaMismatchError,
    TooBigContentError,
    TooManyColumnsError,
)
from libcommon.parquet_utils import EmptyParquetMetadataError, Indexer, SchemaMismatchError, TooBigRows
from libcommon.storage import StrPath
from libcommon.storage_client import StorageClient
from libcommon.utils import JobInfo, Row, RowItem
from libcommon.viewer_utils.features import get_cell_value, to_features_list

from worker.config import AppConfig, FirstRowsConfig
from worker.dtos import CompleteJobResult, SplitFirstRowsResponse
from worker.job_runners.split.split_job_runner import SplitJobRunner
from worker.utils import create_truncated_row_items, get_json_size


def transform_rows(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    rows: list[RowItem],
    features: Features,
    storage_client: StorageClient,
) -> list[Row]:
    return [
        {
            featureName: get_cell_value(
                dataset=dataset,
                revision=revision,
                config=config,
                split=split,
                row_idx=row_idx,
                cell=row["row"][featureName] if featureName in row["row"] else None,
                featureName=featureName,
                fieldType=fieldType,
                storage_client=storage_client,
            )
            for (featureName, fieldType) in features.items()
        }
        for row_idx, row in enumerate(rows)
    ]


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
    logging.info(f"get first-rows for dataset={dataset} config={config} split={split}")

    try:
        rows_index = indexer.get_rows_index(
            dataset=dataset,
            config=config,
            split=split,
        )
    except EmptyParquetMetadataError:
        raise ParquetResponseEmptyError("No parquet files found.")

    # validate the features
    features = rows_index.parquet_index.features
    if features and len(features) > columns_max_number:
        raise TooManyColumnsError(
            f"The number of columns ({len(features)}) exceeds the maximum supported number of columns"
            f" ({columns_max_number}). This is a current limitation of the datasets viewer. You can reduce the number"
            " of columns if you want the viewer to work."
        )

    # validate size of response without the rows
    features_list = to_features_list(features=features)
    response_features_only: SplitFirstRowsResponse = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "features": features_list,
        "rows": [],
        "truncated": False,
    }

    surrounding_json_size = get_json_size(response_features_only)
    if surrounding_json_size > rows_max_bytes:
        raise TooBigContentError(
            f"The size of the content of the first rows ({surrounding_json_size}) exceeds the maximum"
            f" supported size ({rows_max_bytes} B) even after truncation. Please report the issue."
        )

    # get the rows
    try:
        pa_table = rows_index.query(offset=0, length=rows_max_number)
        all_fetched = rows_index.parquet_index.num_rows_total <= rows_max_number
    except TooBigRows as err:
        raise TooBigContentError(str(err))
    except SchemaMismatchError as err:
        raise SplitParquetSchemaMismatchError(
            "Split parquet files being processed have different schemas. Ensure all files have identical column names.",
            cause=err,
        )

    rows = [
        RowItem(
            {
                "row_idx": idx,
                "row": row,
                "truncated_cells": [],
            }
        )
        for idx, row in enumerate(pa_table.to_pylist())
    ]

    # transform the rows, if needed (e.g. save the images or audio to the assets, and return their URL)
    try:
        transformed_rows = transform_rows(
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            rows=rows,
            features=features,
            storage_client=storage_client,
        )
    except Exception as err:
        raise RowsPostProcessingError(
            "Server error while post-processing the split rows. Please report the issue.",
            cause=err,
        ) from err

    # truncate the rows to fit within the restrictions, and prepare them as RowItems
    columns_to_keep_untruncated = [col for col, feature in features.items() if isinstance(feature, (Image, Audio))]
    row_items, truncated = create_truncated_row_items(
        rows=transformed_rows,
        min_cell_bytes=min_cell_bytes,
        rows_max_bytes=rows_max_bytes - surrounding_json_size,
        rows_min_number=rows_min_number,
        columns_to_keep_untruncated=columns_to_keep_untruncated,
    )

    response = response_features_only
    response["rows"] = row_items
    response["truncated"] = (not all_fetched) or truncated

    return response


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
                rows_max_number=self.first_rows_config.max_number,
                rows_min_number=self.first_rows_config.min_number,
                columns_max_number=self.first_rows_config.columns_max_number,
                indexer=self.indexer,
            )
        )
