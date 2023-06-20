# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from pathlib import Path
from typing import List, Optional, Set

import duckdb
from huggingface_hub._commit_api import (
    CommitOperation,
    CommitOperationAdd,
    CommitOperationDelete,
)
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils._errors import RepositoryNotFoundError
from libcommon.config import DuckDbIndexConfig
from libcommon.constants import PROCESSING_STEP_SPLIT_DUCKDB_INDEX_VERSION
from libcommon.exceptions import (
    CachedDirectoryNotInitializedError,
    DatasetNotFoundError,
    LockedDatasetTimeoutError,
    NoIndexableColumnsError,
    NotAvailableIndexFileError,
    ParquetResponseEmptyError,
    PreviousStepFormatError,
    SplitNotFoundError,
    SplitWithTooBigParquetError,
)
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import lock
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.storage import StrPath
from libcommon.utils import JobInfo, SplitHubFile

from worker.config import AppConfig
from worker.job_runners.split.split_job_runner import SplitJobRunnerWithCache
from worker.utils import CompleteJobResult, create_branch, hf_hub_url

DATASET_TYPE = "dataset"
STRING_FEATURE_DTYPE = "string"
VALUE_FEATURE_TYPE = "Value"
DUCKDB_DEFAULT_INDEX_FILENAME = "index.duckdb"
CREATE_SEQUENCE_COMMAND = "CREATE OR REPLACE SEQUENCE serial START 1;"
CREATE_INDEX_COMMAND = "PRAGMA create_fts_index('data', '__hf_index_id', '*', overwrite=1);"
CREATE_TABLE_COMMAND = "CREATE OR REPLACE TABLE data AS SELECT nextval('serial') AS __hf_index_id, {columns} FROM"
INSTALL_EXTENSION_COMMAND = "INSTALL '{extension}';"
LOAD_EXTENSION_COMMAND = "LOAD '{extension}';"


def compute_index_rows(
    job_id: str,
    dataset: str,
    config: str,
    split: str,
    duckdb_index_file_directory: Optional[Path],
    target_revision: str,
    hf_endpoint: str,
    commit_message: str,
    url_template: str,
    hf_token: Optional[str],
    max_parquet_size_bytes: int,
    committer_hf_token: Optional[str],
) -> SplitHubFile:
    logging.info(f"get split-duckdb-index for dataset={dataset} config={config} split={split}")

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

    # get parquet urls and dataset_info
    config_parquet_and_info_step = "config-parquet-and-info"
    parquet_and_info_best_response = get_previous_step_or_raise(
        kinds=[config_parquet_and_info_step],
        dataset=dataset,
        config=config,
    )
    content_parquet_and_info = parquet_and_info_best_response.response["content"]
    try:
        split_parquet_files = [
            parquet_file
            for parquet_file in content_parquet_and_info["parquet_files"]
            if parquet_file["config"] == config and parquet_file["split"] == split
        ]

        split_parquets_size = sum(parquet_file["size"] for parquet_file in split_parquet_files)

        if split_parquets_size > max_parquet_size_bytes:
            raise SplitWithTooBigParquetError(
                f"The indexing is limited to split parquets under {max_parquet_size_bytes} bytes. "
                f"Current size of sum of split parquets is {split_parquets_size} bytes."
            )

        parquet_urls = [parquet_file["url"] for parquet_file in split_parquet_files]

        if not parquet_urls:
            raise ParquetResponseEmptyError("No parquet files found.")

        # get the features
        features = content_parquet_and_info["dataset_info"]["features"]
        column_names = ",".join(list(features.keys()))

        # look for string columns
        string_columns = [
            column
            for column, feature in features.items()
            if "dtype" in feature
            and "_type" in feature
            and feature["dtype"] == STRING_FEATURE_DTYPE
            and feature["_type"] == VALUE_FEATURE_TYPE
        ]
        if not string_columns:
            raise NoIndexableColumnsError("No string columns available to index.")

    except KeyError as e:
        raise PreviousStepFormatError(
            f"Previous step '{config_parquet_and_info_step}' did not return the expected content.", e
        ) from e

    # configure duckdb extensions
    duckdb.execute(INSTALL_EXTENSION_COMMAND.format(extension="httpfs"))
    duckdb.execute(LOAD_EXTENSION_COMMAND.format(extension="httpfs"))
    duckdb.execute(INSTALL_EXTENSION_COMMAND.format(extension="fts"))
    duckdb.execute(LOAD_EXTENSION_COMMAND.format(extension="fts"))

    # index all columns
    if duckdb_index_file_directory is None:
        raise CachedDirectoryNotInitializedError("Cache directory has not been initialized.")
    db_path = Path(duckdb_index_file_directory.resolve() / DUCKDB_DEFAULT_INDEX_FILENAME)

    con = duckdb.connect(str(db_path.resolve()))
    logging.debug(CREATE_SEQUENCE_COMMAND)
    con.sql(CREATE_SEQUENCE_COMMAND)

    create_command_sql = f"{CREATE_TABLE_COMMAND.format(columns=column_names)} read_parquet({parquet_urls});"
    logging.debug(create_command_sql)
    con.sql(create_command_sql)

    # TODO: by default, 'porter' stemmer is being used, use a specific one by dataset language in the future
    # see https://duckdb.org/docs/extensions/full_text_search.html for more details about 'stemmer' parameter
    logging.debug(CREATE_INDEX_COMMAND)
    con.sql(CREATE_INDEX_COMMAND)
    con.close()
    
    # create the target revision if it does not exist yet (clone from initial commit to avoid cloning all repo's files)
    hf_api = HfApi(endpoint=hf_endpoint, token=hf_token)
    committer_hf_api = HfApi(endpoint=hf_endpoint, token=committer_hf_token)
    index_file_location = f"{config}/{split}/{DUCKDB_DEFAULT_INDEX_FILENAME}"
    try:
        refs = hf_api.list_repo_refs(repo_id=dataset, repo_type=DATASET_TYPE)
    except RepositoryNotFoundError as err:
        raise DatasetNotFoundError("The dataset does not exist on the Hub.") from err

    create_branch(
        dataset=dataset, target_revision=target_revision, refs=refs, hf_api=hf_api, committer_hf_api=committer_hf_api
    )

    try:
        sleeps = [1, 1, 1, 10, 10, 100, 100, 100, 300]
        with lock.git_branch(dataset=dataset, branch=target_revision, job_id=job_id, sleeps=sleeps):
            target_dataset_info = hf_api.dataset_info(repo_id=dataset, revision=target_revision, files_metadata=False)
            all_repo_files: Set[str] = {f.rfilename for f in target_dataset_info.siblings}
            delete_operations: List[CommitOperation] = []
            if index_file_location in all_repo_files:
                delete_operations.append(CommitOperationDelete(path_in_repo=index_file_location))

            # send the files to the target revision
            add_operations: List[CommitOperation] = [
                CommitOperationAdd(path_in_repo=index_file_location, path_or_fileobj=db_path.resolve())
            ]

            committer_hf_api.create_commit(
                repo_id=dataset,
                repo_type=DATASET_TYPE,
                revision=target_revision,
                operations=delete_operations + add_operations,
                commit_message=commit_message,
                parent_commit=target_dataset_info.sha,
            )

            # call the API again to get the index file
            target_dataset_info = hf_api.dataset_info(repo_id=dataset, revision=target_revision, files_metadata=True)
    except TimeoutError as err:
        raise LockedDatasetTimeoutError("the dataset is currently locked, please try again later.") from err

    repo_files = [
        repo_file for repo_file in target_dataset_info.siblings if repo_file.rfilename == index_file_location
    ]

    if not repo_files or len(repo_files) != 1:
        logging.warning(f"Found {len(repo_files)} index files, should be only 1")
        raise NotAvailableIndexFileError("No index file was found")

    repo_file = repo_files[0]
    if repo_file.size is None:
        raise ValueError(f"Cannot get size of {repo_file.rfilename}")

    return SplitHubFile(
        dataset=dataset,
        config=config,
        split=split,
        url=hf_hub_url(
            repo_id=dataset,
            filename=repo_file.rfilename,
            hf_endpoint=hf_endpoint,
            revision=target_revision,
            url_template=url_template,
        ),
        filename=Path(repo_file.rfilename).name,
        size=repo_file.size,
    )


class SplitDuckDbIndexJobRunner(SplitJobRunnerWithCache):
    duckdb_index_config: DuckDbIndexConfig

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
        duckdb_index_cache_directory: StrPath,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
            cache_directory=Path(duckdb_index_cache_directory),
        )
        self.duckdb_index_config = app_config.duckdb_index

    @staticmethod
    def get_job_type() -> str:
        return "split-duckdb-index"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_SPLIT_DUCKDB_INDEX_VERSION

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_index_rows(
                job_id=self.job_info["job_id"],
                dataset=self.dataset,
                config=self.config,
                split=self.split,
                duckdb_index_file_directory=self.cache_subdirectory,
                hf_token=self.app_config.common.hf_token,
                url_template=self.duckdb_index_config.url_template,
                commit_message=self.duckdb_index_config.commit_message,
                committer_hf_token=self.duckdb_index_config.committer_hf_token,
                hf_endpoint=self.app_config.common.hf_endpoint,
                target_revision=self.duckdb_index_config.target_revision,
                max_parquet_size_bytes=self.duckdb_index_config.max_parquet_size_bytes,
            )
        )
