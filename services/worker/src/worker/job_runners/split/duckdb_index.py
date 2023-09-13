# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import copy
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Set

import duckdb
from datasets.features.features import Features, FeatureType, Value, _visit
from huggingface_hub import hf_hub_download
from huggingface_hub._commit_api import (
    CommitOperation,
    CommitOperationAdd,
    CommitOperationDelete,
)
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils._errors import HfHubHTTPError, RepositoryNotFoundError
from libcommon.constants import PROCESSING_STEP_SPLIT_DUCKDB_INDEX_VERSION
from libcommon.exceptions import (
    CacheDirectoryNotInitializedError,
    CreateCommitError,
    DatasetNotFoundError,
    DuckDBIndexFileNotFoundError,
    LockedDatasetTimeoutError,
    NoIndexableColumnsError,
    ParquetResponseEmptyError,
    PreviousStepFormatError,
    SplitWithTooBigParquetError,
)
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import lock
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.storage import StrPath
from libcommon.utils import JobInfo, SplitHubFile

from worker.config import AppConfig, DuckDbIndexConfig
from worker.dtos import CompleteJobResult
from worker.job_runners.split.split_job_runner import SplitJobRunnerWithCache
from worker.utils import (
    HF_HUB_HTTP_ERROR_RETRY_SLEEPS,
    LOCK_GIT_BRANCH_RETRY_SLEEPS,
    create_branch,
    hf_hub_url,
    retry,
)

DATASET_TYPE = "dataset"
STRING_FEATURE_DTYPE = "string"
VALUE_FEATURE_TYPE = "Value"
DUCKDB_DEFAULT_INDEX_FILENAME = "index.duckdb"
CREATE_SEQUENCE_COMMAND = "CREATE OR REPLACE SEQUENCE serial START 0 MINVALUE 0;"
CREATE_INDEX_COMMAND = "PRAGMA create_fts_index('data', '__hf_index_id', {columns}, overwrite=1);"
CREATE_TABLE_COMMAND = "CREATE OR REPLACE TABLE data AS SELECT nextval('serial') AS __hf_index_id, {columns} FROM"
INSTALL_EXTENSION_COMMAND = "INSTALL '{extension}';"
LOAD_EXTENSION_COMMAND = "LOAD '{extension}';"
SET_EXTENSIONS_DIRECTORY_COMMAND = "SET extension_directory='{directory}';"
REPO_TYPE = "dataset"
HUB_DOWNLOAD_CACHE_FOLDER = "cache"


class DuckdbIndexWithFeatures(SplitHubFile):
    features: Optional[dict[str, Any]]


def get_indexable_columns(features: Features) -> List[str]:
    indexable_columns: List[str] = []
    for column, feature in features.items():
        indexable = False

        def check_indexable(feature: FeatureType) -> None:
            nonlocal indexable
            if isinstance(feature, Value) and feature.dtype == "string":
                indexable = True

        _visit(feature, check_indexable)
        if indexable:
            indexable_columns.append(column)
    return indexable_columns


def compute_index_rows(
    job_id: str,
    dataset: str,
    config: str,
    split: str,
    duckdb_index_file_directory: Path,
    target_revision: str,
    hf_endpoint: str,
    commit_message: str,
    url_template: str,
    hf_token: Optional[str],
    max_parquet_size_bytes: int,
    extensions_directory: Optional[str],
    committer_hf_token: Optional[str],
) -> DuckdbIndexWithFeatures:
    logging.info(f"get split-duckdb-index for dataset={dataset} config={config} split={split}")

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

        parquet_file_names = [parquet_file["filename"] for parquet_file in split_parquet_files]
        if not parquet_file_names:
            raise ParquetResponseEmptyError("No parquet files found.")

        # For directories like "partial-train" for the file at "en/partial-train/0000.parquet" in the C4 dataset.
        # Note that "-" is forbidden for split names so it doesn't create directory names collisions.
        split_directory = split_parquet_files[0]["url"].rsplit("/", 2)[1]

        # get the features
        features = content_parquet_and_info["dataset_info"]["features"]
        column_names = ",".join('"' + column + '"' for column in list(features.keys()))

        # look for indexable columns (= possibly nested columns containing string data)
        # copy the features is needed but will be fixed with https://github.com/huggingface/datasets/pull/6189
        indexable_columns = ",".join(
            '"' + column + '"' for column in get_indexable_columns(Features.from_dict(copy.deepcopy(features)))
        )
        if not indexable_columns:
            raise NoIndexableColumnsError("No string columns available to index.")

    except KeyError as e:
        raise PreviousStepFormatError(
            f"Previous step '{config_parquet_and_info_step}' did not return the expected content.", e
        ) from e

    # index all columns
    db_path = duckdb_index_file_directory.resolve() / DUCKDB_DEFAULT_INDEX_FILENAME
    con = duckdb.connect(str(db_path.resolve()))

    # configure duckdb extensions
    if extensions_directory is not None:
        con.execute(SET_EXTENSIONS_DIRECTORY_COMMAND.format(directory=extensions_directory))

    con.execute(INSTALL_EXTENSION_COMMAND.format(extension="httpfs"))
    con.execute(LOAD_EXTENSION_COMMAND.format(extension="httpfs"))
    con.execute(INSTALL_EXTENSION_COMMAND.format(extension="fts"))
    con.execute(LOAD_EXTENSION_COMMAND.format(extension="fts"))

    logging.debug(CREATE_SEQUENCE_COMMAND)
    con.sql(CREATE_SEQUENCE_COMMAND)

    # see https://pypi.org/project/hf-transfer/ for more details about how to enable hf_transfer
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    for parquet_file in parquet_file_names:
        hf_hub_download(
            repo_type=REPO_TYPE,
            revision=target_revision,
            repo_id=dataset,
            filename=f"{config}/{split_directory}/{parquet_file}",
            local_dir=duckdb_index_file_directory,
            local_dir_use_symlinks=False,
            token=hf_token,
            cache_dir=duckdb_index_file_directory,
        )

    all_split_parquets = f"{duckdb_index_file_directory}/{config}/{split_directory}/*.parquet"
    create_command_sql = f"{CREATE_TABLE_COMMAND.format(columns=column_names)} '{all_split_parquets}';"
    logging.debug(create_command_sql)
    con.sql(create_command_sql)

    # TODO: by default, 'porter' stemmer is being used, use a specific one by dataset language in the future
    # see https://duckdb.org/docs/extensions/full_text_search.html for more details about 'stemmer' parameter
    create_index_sql = CREATE_INDEX_COMMAND.format(columns=indexable_columns)
    logging.debug(create_index_sql)
    con.sql(create_index_sql)
    con.close()

    hf_api = HfApi(endpoint=hf_endpoint, token=hf_token)
    committer_hf_api = HfApi(endpoint=hf_endpoint, token=committer_hf_token)
    index_file_location = f"{config}/{split_directory}/{DUCKDB_DEFAULT_INDEX_FILENAME}"

    try:
        with lock.git_branch(
            dataset=dataset, branch=target_revision, owner=job_id, sleeps=LOCK_GIT_BRANCH_RETRY_SLEEPS
        ):
            logging.debug(f"try to create branch for {dataset=} with {target_revision=} on {hf_endpoint=}")
            create_branch(
                dataset=dataset,
                target_revision=target_revision,
                hf_api=hf_api,
                committer_hf_api=committer_hf_api,
            )

            logging.debug(f"get dataset info for {dataset=} with {target_revision=}")
            target_dataset_info = hf_api.dataset_info(repo_id=dataset, revision=target_revision, files_metadata=False)
            all_repo_files: Set[str] = {f.rfilename for f in target_dataset_info.siblings}
            delete_operations: List[CommitOperation] = []
            if index_file_location in all_repo_files:
                delete_operations.append(CommitOperationDelete(path_in_repo=index_file_location))
            logging.debug(f"delete operations for {dataset=} {delete_operations=}")

            # send the files to the target revision
            add_operations: List[CommitOperation] = [
                CommitOperationAdd(path_in_repo=index_file_location, path_or_fileobj=db_path.resolve())
            ]
            logging.debug(f"add operations for {dataset=} {add_operations=}")

            retry_create_commit = retry(on=[HfHubHTTPError], sleeps=HF_HUB_HTTP_ERROR_RETRY_SLEEPS)(
                committer_hf_api.create_commit
            )
            try:
                retry_create_commit(
                    repo_id=dataset,
                    repo_type=DATASET_TYPE,
                    revision=target_revision,
                    operations=delete_operations + add_operations,
                    commit_message=commit_message,
                    parent_commit=target_dataset_info.sha,
                )
            except RuntimeError as e:
                if e.__cause__ and isinstance(e.__cause__, HfHubHTTPError):
                    raise CreateCommitError(
                        message=(
                            f"Commit {commit_message} could not be created on the Hub (after"
                            f" {len(HF_HUB_HTTP_ERROR_RETRY_SLEEPS)} attempts)."
                        ),
                        cause=e.__cause__,
                    ) from e.__cause__
                raise e

            logging.debug(f"create commit {commit_message} for {dataset=} {add_operations=}")

            # call the API again to get the index file
            target_dataset_info = hf_api.dataset_info(repo_id=dataset, revision=target_revision, files_metadata=True)
            logging.debug(f"dataset info for {dataset=} {target_dataset_info=}")
    except TimeoutError as err:
        raise LockedDatasetTimeoutError("the dataset is currently locked, please try again later.") from err
    except RepositoryNotFoundError as err:
        raise DatasetNotFoundError("The dataset does not exist on the Hub.") from err

    repo_files = [
        repo_file for repo_file in target_dataset_info.siblings if repo_file.rfilename == index_file_location
    ]

    if not repo_files or len(repo_files) != 1:
        logging.warning(f"Found {len(repo_files)} index files, should be only 1")
        raise DuckDBIndexFileNotFoundError("No index file was found")

    repo_file = repo_files[0]
    if repo_file.size is None:
        raise ValueError(f"Cannot get size of {repo_file.rfilename}")

    # we added the __hf_index_id column for the index
    features["__hf_index_id"] = {"dtype": "int64", "_type": "Value"}

    return DuckdbIndexWithFeatures(
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
        features=features,
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
        if self.cache_subdirectory is None:
            raise CacheDirectoryNotInitializedError("Cache directory has not been initialized.")
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
                extensions_directory=self.duckdb_index_config.extensions_directory,
                committer_hf_token=self.duckdb_index_config.committer_hf_token,
                hf_endpoint=self.app_config.common.hf_endpoint,
                target_revision=self.duckdb_index_config.target_revision,
                max_parquet_size_bytes=self.duckdb_index_config.max_parquet_size_bytes,
            )
        )
