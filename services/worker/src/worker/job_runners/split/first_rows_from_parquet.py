# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from datasets import Audio, Features, Image
from fsspec.implementations.http import HTTPFileSystem
from libcommon.constants import (
    PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_PARQUET_VERSION,
    PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_STREAMING_VERSION,
)
from libcommon.exceptions import (
    RowsPostProcessingError,
    TooBigContentError,
    TooManyColumnsError,
)
from libcommon.parquet_utils import Indexer, TooBigRows
from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.s3_client import S3Client
from libcommon.storage import StrPath
from libcommon.storage_options import S3StorageOptions
from libcommon.utils import JobInfo, Row, RowItem
from libcommon.viewer_utils.features import get_cell_value, to_features_list

from worker.config import AppConfig, FirstRowsConfig
from worker.dtos import CompleteJobResult, JobRunnerInfo, SplitFirstRowsResponse
from worker.job_runners.split.split_job_runner import SplitJobRunner
from worker.utils import create_truncated_row_items, get_json_size


def transform_rows(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    rows: list[RowItem],
    features: Features,
    storage_options: S3StorageOptions,
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
                storage_options=storage_options,
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
    storage_options: S3StorageOptions,
    min_cell_bytes: int,
    rows_max_bytes: int,
    rows_max_number: int,
    rows_min_number: int,
    columns_max_number: int,
    indexer: Indexer,
) -> SplitFirstRowsResponse:
    logging.info(f"get first-rows for dataset={dataset} config={config} split={split}")

    rows_index = indexer.get_rows_index(
        dataset=dataset,
        config=config,
        split=split,
    )

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
            storage_options=storage_options,
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
    assets_directory: StrPath
    first_rows_config: FirstRowsConfig
    indexer: Indexer

    @staticmethod
    def get_job_type() -> str:
        return "split-first-rows-from-parquet"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_PARQUET_VERSION

    @staticmethod
    def get_parallel_job_runner() -> JobRunnerInfo:
        return JobRunnerInfo(
            job_runner_version=PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_STREAMING_VERSION,
            job_type="split-first-rows-from-streaming",
        )

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
        processing_graph: ProcessingGraph,
        assets_directory: StrPath,
        parquet_metadata_directory: StrPath,
        s3_client: S3Client,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
        )
        self.first_rows_config = app_config.first_rows
        self.assets_directory = assets_directory
        self.assets_base_url = app_config.assets.base_url
        self.parquet_metadata_directory = parquet_metadata_directory
        self.indexer = Indexer(
            processing_graph=processing_graph,
            hf_token=self.app_config.common.hf_token,
            parquet_metadata_directory=parquet_metadata_directory,
            httpfs=HTTPFileSystem(headers={"authorization": f"Bearer {self.app_config.common.hf_token}"}),
            unsupported_features=[],
            all_columns_supported_datasets_allow_list="all",
            max_arrow_data_in_memory=app_config.rows_index.max_arrow_data_in_memory,
        )

        self.storage_options = S3StorageOptions(
            assets_base_url=self.assets_base_url,
            assets_directory=self.assets_directory,
            overwrite=True,
            s3_client=s3_client,
            s3_folder_name=app_config.assets.s3_folder_name,
        )

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_first_rows_response(
                dataset=self.dataset,
                revision=self.dataset_git_revision,
                config=self.config,
                split=self.split,
                storage_options=self.storage_options,
                min_cell_bytes=self.first_rows_config.min_cell_bytes,
                rows_max_bytes=self.first_rows_config.max_bytes,
                rows_max_number=self.first_rows_config.max_number,
                rows_min_number=self.first_rows_config.min_number,
                columns_max_number=self.first_rows_config.columns_max_number,
                indexer=self.indexer,
            )
        )
