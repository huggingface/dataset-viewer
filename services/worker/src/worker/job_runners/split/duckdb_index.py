# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from functools import partial
from typing import List, Optional

import duckdb
from datasets import Features
from libcommon.constants import PROCESSING_STEP_SPLIT_DUCKDB_INDEX_VERSION
from libcommon.exceptions import (
    FileSystemError,
    NoIndexableColumnsError,
    ParquetResponseEmptyError,
    PreviousStepFormatError,
    SplitNotFoundError,
    UnsupportedIndexableColumnsError,
)
from libcommon.processing_graph import ProcessingStep
from libcommon.storage import StrPath
from libcommon.utils import JobInfo
from libcommon.viewer_utils.index_utils import create_index_dir_split
from pyarrow.parquet import ParquetFile
from tqdm.contrib.concurrent import thread_map

from worker.config import AppConfig
from worker.job_runners.split.split_job_runner import SplitJobRunner
from worker.utils import (
    CompleteJobResult,
    IndexRowsResponse,
    get_hf_fs,
    get_hf_parquet_uris,
    get_previous_step_or_raise,
)

STRING_FEATURE_DTYPE = "string"
VALUE_FEATURE_TYPE = "Value"
DUCKDB_DEFAULT_INDEX_FILENAME = "index.db"
UNSUPPORTED_FEATURES_MAGIC_STRINGS = ["'binary'", "Audio", "Image"]
CREATE_SEQUENCE_COMMAND = "CREATE OR REPLACE SEQUENCE serial START 1;"
CREATE_INDEX_COMMAND = "PRAGMA create_fts_index('data', '__id', '*', overwrite=1);"
CREATE_TABLE_COMMAND = "CREATE OR REPLACE TABLE data AS SELECT nextval('serial') AS __id, * FROM"
INSTALL_EXTENSION_COMMAND = "INSTALL '{extension}';"
LOAD_EXTENSION_COMMAND = "LOAD '{extension}';"
# TODO: What if __id field already exist?


def compute_index_rows(
    dataset: str,
    config: str,
    split: str,
    duckdb_index_directory: StrPath,
    hf_token: Optional[str],
) -> IndexRowsResponse:
    logging.info(f"get index-rows for dataset={dataset} config={config} split={split}")

    # validate split
    split_names_best_response = get_previous_step_or_raise(
        kinds=["config-split-names-from-streaming", "config-split-names-from-info"], dataset=dataset, config=config
    )
    try:
        splits_content = split_names_best_response.response["content"]["splits"]
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    if split not in [split_item["split"] for split_item in splits_content]:
        raise SplitNotFoundError(f"The split '{split}' does not exist for the config '{config}' of the dataset.")

    # get parquet content
    config_parquet_best_response = get_previous_step_or_raise(kinds=["config-parquet"], dataset=dataset, config=config)

    try:
        parquet_files_content = config_parquet_best_response.response["content"]["parquet_files"]
        sources = sorted(
            f"{config}/{parquet_file['filename']}"
            for parquet_file in parquet_files_content
            if parquet_file["split"] == split and parquet_file["config"] == config
        )
        if not sources:
            raise ParquetResponseEmptyError("No parquet files found.")
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.") from e

    logging.debug(f"Found {len(sources)} parquet files for {dataset=}, {config=}, {split=}: {sources}")

    fs = get_hf_fs(hf_token=hf_token)
    source_uris = get_hf_parquet_uris(sources, dataset=dataset)
    desc = f"{dataset}/{config}/{split}"
    try:
        parquet_files: List[ParquetFile] = thread_map(
            partial(ParquetFile, filesystem=fs), source_uris, desc=desc, unit="pq", disable=True
        )
    except Exception as e:
        raise FileSystemError(f"Could not read the parquet files: {e}") from e

    # get the features
    features = Features.from_arrow_schema(parquet_files[0].schema.to_arrow_schema())

    # look for string columns using the first rows
    string_columns = [column for column, feature in features.items() if STRING_FEATURE_DTYPE in str(feature)]

    if not string_columns:
        raise NoIndexableColumnsError("No string columns available to index.")

    # look for image, audio and binary columns, if present, raise exeception do not supported yet and index everything
    if any(
        feature
        for feature in features.values()
        if next(
            (feature_type for feature_type in UNSUPPORTED_FEATURES_MAGIC_STRINGS if feature_type in str(feature)), None
        )
        is not None
    ):
        raise UnsupportedIndexableColumnsError("Unsupported feature types for indexing.")

    try:
        parquet_urls = [content["url"] for content in parquet_files_content if content["split"] == split]

        if not parquet_urls:
            raise ParquetResponseEmptyError("No parquet files found.")
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.") from e

    # create duckdb index location
    split_path, dir_path = create_index_dir_split(
        dataset=dataset, config=config, split=split, index_directory=duckdb_index_directory
    )
    duckdb_index_filename = f"{split_path}/{DUCKDB_DEFAULT_INDEX_FILENAME}"
    db_location = dir_path / DUCKDB_DEFAULT_INDEX_FILENAME

    # configure duckdb extensions
    duckdb.execute(INSTALL_EXTENSION_COMMAND.format(extension="httpfs"))
    duckdb.execute(LOAD_EXTENSION_COMMAND.format(extension="httpfs"))
    duckdb.execute(INSTALL_EXTENSION_COMMAND.format(extension="fts"))
    duckdb.execute(LOAD_EXTENSION_COMMAND.format(extension="fts"))

    # index
    con = duckdb.connect(str(db_location))
    con.sql(CREATE_SEQUENCE_COMMAND)
    con.sql(f"{CREATE_TABLE_COMMAND} read_parquet({parquet_urls});")

    # TODO: by default, 'porter' stemmer is being used, use a specific one by dataset language in the future
    # see https://duckdb.org/docs/extensions/full_text_search.html for more deails about 'stemmer' parameter
    con.sql(CREATE_INDEX_COMMAND)

    return IndexRowsResponse(duckdb_index_filename=duckdb_index_filename)


class SplitDuckDbIndexJobRunner(SplitJobRunner):
    duckdb_index_directory: StrPath

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
        duckdb_index_directory: StrPath,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
        )
        self.duckdb_index_directory = duckdb_index_directory

    @staticmethod
    def get_job_type() -> str:
        return "split-duckdb-index"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_SPLIT_DUCKDB_INDEX_VERSION

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_index_rows(
                dataset=self.dataset,
                config=self.config,
                split=self.split,
                duckdb_index_directory=self.duckdb_index_directory,
                hf_token=self.app_config.common.hf_token,
            )
        )
