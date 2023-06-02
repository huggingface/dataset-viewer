# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

import duckdb
from libcommon.constants import PROCESSING_STEP_SPLIT_DUCKDB_INDEX_VERSION
from libcommon.exceptions import (
    NoIndexableColumnsError,
    ParquetResponseEmptyError,
    PreviousStepFormatError,
    UnsupportedIndexableColumnsError,
)
from libcommon.processing_graph import ProcessingStep
from libcommon.storage import StrPath
from libcommon.utils import JobInfo
from libcommon.viewer_utils.index_utils import create_index_dir_split

from worker.config import AppConfig
from worker.job_runners.split.split_job_runner import SplitJobRunner
from worker.utils import (
    CompleteJobResult,
    IndexRowsResponse,
    get_previous_step_or_raise,
)

STRING_FEATURE_DTYPE = "string"
VALUE_FEATURE_TYPE = "Value"
DUCKDB_DEFAULT_DB_NAME = "index.db"
UNSUPPORTED_FEATURES_MAGIC_STRINGS = ["'binary'", "Audio", "Image"]
CREATE_SEQUENCE_COMMAND = "CREATE OR REPLACE SEQUENCE serial START 1;"
DROP_INDEX_COMMAND = "PRAGMA drop_fts_index('data');"
CREATE_INDEX_COMMAND = "PRAGMA create_fts_index('data', '__id', '*');"
CREATE_TABLE_COMMAND = "CREATE OR REPLACE TABLE data AS SELECT nextval('serial') AS __id, * FROM"
INSTALL_EXTENSION_COMMAND = "INSTALL '{extension}';"
LOAD_EXTENSION_COMMAND = "LOAD '{extension}';"
# TODO: What if __id field already exist?


def compute_index_rows(dataset: str, config: str, split: str, duckdb_index_directory: StrPath) -> IndexRowsResponse:
    logging.info(f"get index-rows for dataset={dataset} config={config} split={split}")

    # get the first rows from previous job
    upstream_response = get_previous_step_or_raise(
        kinds=["split-first-rows-from-streaming", "split-first-rows-from-parquet"],
        dataset=dataset,
        config=config,
        split=split,
    )
    try:
        first_rows = upstream_response.response["content"]
        features = first_rows["features"]
    except KeyError as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    # look for string columns using the first rows
    string_columns = [
        feature["name"]
        for feature in features
        if "dtype" in feature["type"]
        and "_type" in feature["type"]
        and feature["type"]["dtype"] == STRING_FEATURE_DTYPE
        and feature["type"]["_type"] == VALUE_FEATURE_TYPE
    ]

    if not string_columns:
        raise NoIndexableColumnsError("No string columns available to index.")

    # look for image, audio and binary columns, if present, raise exeception do not supported yet and index everything
    if any(
        feature["name"]
        for feature in features
        if "_type" in feature["type"] and feature["type"]["_type"] in UNSUPPORTED_FEATURES_MAGIC_STRINGS
    ):
        raise UnsupportedIndexableColumnsError("Unsupported feature types for indexing.")

    # get list of parquet urls
    config_parquet = get_previous_step_or_raise(kinds=["config-parquet"], dataset=dataset, config=config)
    try:
        parquet_files = config_parquet.response["content"]["parquet_files"]
        parquet_urls = [content["url"] for content in parquet_files if content["split"] == split]

        if not parquet_urls:
            raise ParquetResponseEmptyError("No parquet files found.")
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.") from e

    # create duckdb index location
    split_path, dir_path = create_index_dir_split(
        dataset=dataset, config=config, split=split, index_directory=duckdb_index_directory
    )
    duck_db_name = f"{split_path}/{DUCKDB_DEFAULT_DB_NAME}"
    db_location = dir_path / DUCKDB_DEFAULT_DB_NAME

    # configure duckdb extensions
    duckdb.execute(INSTALL_EXTENSION_COMMAND.format(extension="httpfs"))
    duckdb.execute(LOAD_EXTENSION_COMMAND.format(extension="httpfs"))
    duckdb.execute(INSTALL_EXTENSION_COMMAND.format(extension="fts"))
    duckdb.execute(LOAD_EXTENSION_COMMAND.format(extension="fts"))

    # index
    con = duckdb.connect(str(db_location))
    con.sql(CREATE_SEQUENCE_COMMAND)
    con.sql(f"{CREATE_TABLE_COMMAND} read_parquet({parquet_urls});")
    con.sql(DROP_INDEX_COMMAND)

    # TODO: by default, 'porter' stemmer is being used, use a specific one by dataset language in the future
    # see https://duckdb.org/docs/extensions/full_text_search.html for more deails about 'stemmer' parameter
    con.sql(CREATE_INDEX_COMMAND)

    return IndexRowsResponse(duckdb_db_name=duck_db_name)


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
            )
        )
