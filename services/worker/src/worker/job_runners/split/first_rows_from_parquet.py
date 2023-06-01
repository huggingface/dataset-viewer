# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import List

from datasets import Features
from libcommon.constants import (
    PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_PARQUET_VERSION,
    PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_STREAMING_VERSION,
)
from libcommon.exceptions import (
    RowsPostProcessingError,
    TooBigContentError,
    TooManyColumnsError,
)
from libcommon.parquet import Indexer
from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.storage import StrPath
from libcommon.utils import JobInfo
from libcommon.viewer_utils.features import get_cell_value

from worker.config import AppConfig, FirstRowsConfig
from worker.job_runners.split.split_job_runner import SplitJobRunner
from worker.utils import (
    CompleteJobResult,
    JobRunnerInfo,
    Row,
    RowItem,
    SplitFirstRowsResponse,
    create_truncated_row_items,
    get_json_size,
    to_features_list,
)


def transform_rows(
    dataset: str,
    config: str,
    split: str,
    rows: List[RowItem],
    features: Features,
    assets_base_url: str,
    assets_directory: StrPath,
) -> List[Row]:
    return [
        {
            featureName: get_cell_value(
                dataset=dataset,
                config=config,
                split=split,
                row_idx=row_idx,
                cell=row["row"][featureName] if featureName in row["row"] else None,
                featureName=featureName,
                fieldType=fieldType,
                assets_base_url=assets_base_url,
                assets_directory=assets_directory,
            )
            for (featureName, fieldType) in features.items()
        }
        for row_idx, row in enumerate(rows)
    ]


def compute_first_rows_response(
    dataset: str,
    config: str,
    split: str,
    assets_base_url: str,
    min_cell_bytes: int,
    rows_max_bytes: int,
    rows_max_number: int,
    rows_min_number: int,
    columns_max_number: int,
    assets_directory: StrPath,
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
    }

    surrounding_json_size = get_json_size(response_features_only)
    if surrounding_json_size > rows_max_bytes:
        raise TooBigContentError(
            f"The size of the content of the first rows ({surrounding_json_size}) exceeds the maximum"
            f" supported size ({rows_max_bytes} B) even after truncation. Please report the issue."
        )

    # get the rows
    pa_table = rows_index.query(offset=0, length=rows_max_number)
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
            config=config,
            split=split,
            rows=rows,
            features=features,
            assets_base_url=assets_base_url,
            assets_directory=assets_directory,
        )
    except Exception as err:
        raise RowsPostProcessingError(
            "Server error while post-processing the split rows. Please report the issue.",
            cause=err,
        ) from err

    # truncate the rows to fit within the restrictions, and prepare them as RowItems
    row_items = create_truncated_row_items(
        rows=transformed_rows,
        min_cell_bytes=min_cell_bytes,
        rows_max_bytes=rows_max_bytes - surrounding_json_size,
        rows_min_number=rows_min_number,
    )

    response = response_features_only
    response["rows"] = row_items
    return response


class SplitFirstRowsFromParquetJobRunner(SplitJobRunner):
    assets_directory: StrPath
    first_rows_config: FirstRowsConfig
    indexed: Indexer

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
            job_type="config-split-names-from-info",
        )

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
        processing_graph: ProcessingGraph,
        assets_directory: StrPath,
        parquet_metadata_directory: StrPath,
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
            unsupported_features_magic_strings=[],
            all_columns_supported_datasets_allow_list="all",
        )

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_first_rows_response(
                dataset=self.dataset,
                config=self.config,
                split=self.split,
                assets_base_url=self.assets_base_url,
                assets_directory=self.assets_directory,
                min_cell_bytes=self.first_rows_config.min_cell_bytes,
                rows_max_bytes=self.first_rows_config.max_bytes,
                rows_max_number=self.first_rows_config.max_number,
                rows_min_number=self.first_rows_config.min_number,
                columns_max_number=self.first_rows_config.columns_max_number,
                indexer=self.indexer,
            )
        )
