# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import copy
import logging
import re
from pathlib import Path
from typing import Any, Literal, Optional

import duckdb
import polars as pl
from datasets.features.features import Features, FeatureType, Translation, TranslationVariableLanguages, Value, _visit
from huggingface_hub._commit_api import (
    CommitOperation,
    CommitOperationAdd,
    CommitOperationDelete,
)
from huggingface_hub.hf_api import HfApi
from huggingface_hub.repocard_data import DatasetCardData
from huggingface_hub.utils._errors import HfHubHTTPError, RepositoryNotFoundError
from libcommon.constants import DUCKDB_INDEX_JOB_RUNNER_SUBDIRECTORY, ROW_IDX_COLUMN
from libcommon.dtos import JobInfo
from libcommon.exceptions import (
    CacheDirectoryNotInitializedError,
    CreateCommitError,
    DatasetNotFoundError,
    DuckDBIndexFileNotFoundError,
    LockedDatasetTimeoutError,
    ParquetResponseEmptyError,
    PreviousStepFormatError,
)
from libcommon.parquet_utils import (
    extract_split_directory_from_parquet_url,
    get_num_parquet_files_to_process,
    is_list_pa_type,
    parquet_export_is_partial,
)
from libcommon.queue.lock import lock
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.storage import StrPath
from libcommon.utils import HF_HUB_HTTP_ERROR_RETRY_SLEEPS, download_file_from_hub, retry

from worker.config import AppConfig, DuckDbIndexConfig
from worker.dtos import CompleteJobResult, SplitDuckdbIndex
from worker.job_runners.split.split_job_runner import SplitJobRunnerWithCache
from worker.statistics_utils import (
    STRING_DTYPES,
    AudioColumn,
    ImageColumn,
    ListColumn,
    StringColumn,
)
from worker.utils import (
    LOCK_GIT_BRANCH_RETRY_SLEEPS,
    create_branch,
    get_split_names,
    hf_hub_url,
)

DATASET_TYPE = "dataset"
DEFAULT_STEMMER = "none"  # Exact word matches
DUCKDB_DEFAULT_INDEX_FILENAME = "index.duckdb"
DUCKDB_DEFAULT_PARTIAL_INDEX_FILENAME = "partial-index.duckdb"
CREATE_INDEX_COMMAND = (
    f"PRAGMA create_fts_index('data', '{ROW_IDX_COLUMN}', {{columns}}, stemmer='{{stemmer}}', overwrite=1);"
)
CREATE_TABLE_COMMAND = "CREATE OR REPLACE TABLE data AS SELECT {columns} FROM '{source}';"
CREATE_TABLE_JOIN_WITH_TRANSFORMED_DATA_COMMAND = """
    CREATE OR REPLACE TABLE data AS 
    SELECT {columns}, transformed_df.* FROM '{source}' 
    POSITIONAL JOIN transformed_df;
"""
CREATE_SEQUENCE_COMMAND = "CREATE OR REPLACE SEQUENCE serial START 0 MINVALUE 0;"
ALTER_TABLE_BY_ADDING_SEQUENCE_COLUMN = (
    f"ALTER TABLE data ADD COLUMN {ROW_IDX_COLUMN} BIGINT DEFAULT nextval('serial');"
)
CREATE_INDEX_ID_COLUMN_COMMANDS = CREATE_SEQUENCE_COMMAND + ALTER_TABLE_BY_ADDING_SEQUENCE_COLUMN
INSTALL_AND_LOAD_EXTENSION_COMMAND = "INSTALL 'fts'; LOAD 'fts';"
SET_EXTENSIONS_DIRECTORY_COMMAND = "SET extension_directory='{directory}';"
REPO_TYPE = "dataset"
# Only some languages are supported, see: https://duckdb.org/docs/extensions/full_text_search.html#pragma-create_fts_index
STEMMER_MAPPING = {
    # Stemmer : ["value ISO 639-1", "value ISO 639-2/3"]
    "arabic": ["ar", "ara"],
    "basque": ["eu", "eus"],
    "catalan": ["ca", "cat"],
    "danish": ["da", "dan"],
    "dutch": ["nl", "nld"],
    "english": ["en", "eng"],
    "finnish": ["fi", "fin"],
    "french": ["fr", "fra"],
    "german": ["de", "deu"],
    "greek": ["el", "ell"],
    "hindi": ["hi", "hin"],
    "hungarian": ["hu", "hun"],
    "indonesian": ["id", "ind"],
    "irish": ["ga", "gle"],
    "italian": ["it", "ita"],
    "lithuanian": ["lt", "lit"],
    "nepali": ["ne", "nep"],
    "norwegian": ["no", "nor"],
    "portuguese": ["pt", "por"],
    "romanian": ["ro", "ron"],
    "russian": ["ru", "rus"],
    "serbian": ["sr", "srp"],
    "spanish": ["es", "spa"],
    "swedish": ["sv", "swe"],
    "tamil": ["ta", "tam"],
    "turkish": ["tr", "tur"],
}

LengthDtype = Literal["string", "list"]


def get_indexable_columns(features: Features) -> list[str]:
    indexable_columns: list[str] = []
    for column, feature in features.items():
        indexable = False

        def check_indexable(feature: FeatureType) -> None:
            nonlocal indexable
            if isinstance(feature, Value) and feature.dtype in STRING_DTYPES:
                indexable = True
            elif isinstance(feature, (Translation, TranslationVariableLanguages)):
                indexable = True

        _visit(feature, check_indexable)
        if indexable:
            indexable_columns.append(column)
    return indexable_columns


def get_monolingual_stemmer(card_data: DatasetCardData) -> str:
    if card_data is None:
        return DEFAULT_STEMMER
    all_languages = card_data["language"]
    if isinstance(all_languages, list) and len(all_languages) == 1:
        first_language = all_languages[0]
    elif isinstance(all_languages, str):
        first_language = all_languages
    else:
        return DEFAULT_STEMMER

    return next((language for language, codes in STEMMER_MAPPING.items() if first_language in codes), DEFAULT_STEMMER)


def compute_length_column(
    parquet_directory: Path,
    column_name: str,
    target_df: Optional[pl.DataFrame],
    dtype: LengthDtype,
) -> pl.DataFrame:
    column_class = ListColumn if dtype == "list" else StringColumn
    df = pl.read_parquet(str(parquet_directory / "*.parquet"), columns=[column_name])
    lengths_column_name = f"{column_name}.length"
    lengths_df: pl.DataFrame = column_class.compute_transformed_data(
        df, column_name, transformed_column_name=lengths_column_name
    )
    if target_df is None:
        return lengths_df.select(pl.col(lengths_column_name))

    target_df.insert_column(target_df.shape[1], lengths_df[lengths_column_name])
    return target_df


def compute_audio_duration_column(
    parquet_directory: Path,
    column_name: str,
    target_df: Optional[pl.DataFrame],
) -> pl.DataFrame:
    duration_column_name = f"{column_name}.duration"
    durations = AudioColumn.compute_transformed_data(parquet_directory, column_name, AudioColumn.get_duration)
    duration_df = pl.from_dict({duration_column_name: durations})
    if target_df is None:
        return duration_df
    target_df.insert_column(target_df.shape[1], duration_df[duration_column_name])
    return target_df


def compute_image_width_height_column(
    parquet_directory: Path,
    column_name: str,
    target_df: Optional[pl.DataFrame],
) -> pl.DataFrame:
    shapes = ImageColumn.compute_transformed_data(parquet_directory, column_name, ImageColumn.get_shape)
    widths, heights = list(zip(*shapes))
    width_column_name, height_column_name = f"{column_name}.width", f"{column_name}.height"
    shapes_df = pl.from_dict({width_column_name: widths, height_column_name: heights})
    if target_df is None:
        return shapes_df
    target_df.insert_column(target_df.shape[1], shapes_df[width_column_name])
    target_df.insert_column(target_df.shape[1], shapes_df[height_column_name])
    return target_df


def compute_transformed_data(parquet_directory: Path, features: dict[str, Any]) -> Optional[pl.DataFrame]:
    transformed_df = None
    for feature_name, feature in features.items():
        if isinstance(feature, list) or (isinstance(feature, dict) and feature.get("_type") == "Sequence"):
            first_parquet_file = list(parquet_directory.glob("*.parquet"))[0]
            if is_list_pa_type(first_parquet_file, feature_name):
                transformed_df = compute_length_column(parquet_directory, feature_name, transformed_df, dtype="list")

        elif isinstance(feature, dict):
            if feature.get("_type") == "Value" and feature.get("dtype") in STRING_DTYPES:
                transformed_df = compute_length_column(parquet_directory, feature_name, transformed_df, dtype="string")

            elif feature.get("_type") == "Audio":
                transformed_df = compute_audio_duration_column(parquet_directory, feature_name, transformed_df)

            elif feature.get("_type") == "Image":
                transformed_df = compute_image_width_height_column(parquet_directory, feature_name, transformed_df)

    return transformed_df


def get_delete_operations(all_repo_files: set[str], split_names: set[str], config: str) -> list[CommitOperationDelete]:
    same_config_pattern = re.compile(f"^({re.escape(config)})/")
    existing_split_pattern = re.compile(
        f"^({'|'.join(re.escape(f'{config}/{split_name}') for split_name in split_names)})/"
    )
    # For directories like "partial-train" for the file at "en/partial-train/0000.parquet" in the C4 dataset.
    # Note that "-" is forbidden for split names so it doesn't create directory names collisions.
    # caveat: the split could become full processed
    existing_partial_split_pattern = re.compile(
        f"^({'|'.join(re.escape(f'{config}/partial-{split_name}') for split_name in split_names)})/"
    )

    return [
        CommitOperationDelete(path_in_repo=file)
        for file in all_repo_files
        if same_config_pattern.match(file)
        and file.endswith(".duckdb")
        and not existing_split_pattern.match(file)
        and not existing_partial_split_pattern.match(file)
    ]


def compute_split_duckdb_index_response(
    job_id: str,
    dataset: str,
    config: str,
    split: str,
    duckdb_index_file_directory: Path,
    target_revision: str,
    source_revision: str,
    hf_endpoint: str,
    commit_message: str,
    url_template: str,
    hf_token: Optional[str],
    max_split_size_bytes: int,
    extensions_directory: Optional[str],
    committer_hf_token: Optional[str],
    parquet_metadata_directory: StrPath,
) -> SplitDuckdbIndex:
    logging.info(f"compute 'split-duckdb-index' for {dataset=} {config=} {split=}")

    # get parquet urls and dataset_info
    config_parquet_metadata_step = "config-parquet-metadata"
    parquet_metadata_response = get_previous_step_or_raise(
        kind=config_parquet_metadata_step,
        dataset=dataset,
        config=config,
    )
    content_parquet_metadata = parquet_metadata_response["content"]
    try:
        split_parquet_files = [
            parquet_file
            for parquet_file in content_parquet_metadata["parquet_files_metadata"]
            if parquet_file["config"] == config and parquet_file["split"] == split
        ]

        if not split_parquet_files:
            raise ParquetResponseEmptyError("No parquet files found.")

        # For directories like "partial-train" for the file at "en/partial-train/0000.parquet" in the C4 dataset.
        # Note that "-" is forbidden for split names so it doesn't create directory names collisions.
        split_directory = extract_split_directory_from_parquet_url(split_parquet_files[0]["url"])
        partial_parquet_export = parquet_export_is_partial(split_parquet_files[0]["url"])

        num_parquet_files_to_index, num_bytes, num_rows = get_num_parquet_files_to_process(
            parquet_files=split_parquet_files,
            parquet_metadata_directory=parquet_metadata_directory,
            max_size_bytes=max_split_size_bytes,
        )

        index_filename = (
            DUCKDB_DEFAULT_PARTIAL_INDEX_FILENAME
            if (num_parquet_files_to_index < len(split_parquet_files))
            else DUCKDB_DEFAULT_INDEX_FILENAME
        )
        partial = partial_parquet_export or (num_parquet_files_to_index < len(split_parquet_files))
        split_parquet_files = split_parquet_files[:num_parquet_files_to_index]
        parquet_file_names = [parquet_file["filename"] for parquet_file in split_parquet_files]

        # get the features
        features = content_parquet_metadata["features"]
        column_names = ",".join(f'"{column}"' for column in features)

        # look for indexable columns (= possibly nested columns containing string data)
        # copy the features is needed but will be fixed with https://github.com/huggingface/datasets/pull/6189
        indexable_columns = ",".join(
            f'"{column}"' for column in get_indexable_columns(Features.from_dict(copy.deepcopy(features)))
        )

    except KeyError as e:
        raise PreviousStepFormatError(
            f"Previous step '{config_parquet_metadata_step}' did not return the expected content.", e
        ) from e

    for parquet_file in parquet_file_names:
        download_file_from_hub(
            repo_type=REPO_TYPE,
            revision=source_revision,
            repo_id=dataset,
            filename=f"{config}/{split_directory}/{parquet_file}",
            local_dir=duckdb_index_file_directory,
            hf_token=hf_token,
            cache_dir=duckdb_index_file_directory,
            force_download=True,
            resume_download=False,
        )
    split_parquet_directory = duckdb_index_file_directory / config / split_directory
    all_split_parquets = str(split_parquet_directory / "*.parquet")

    transformed_df = None
    try:
        transformed_df = compute_transformed_data(split_parquet_directory, features)
    except Exception as err:
        logging.debug(f"Unable to compute transformed data {err}, skipping statistics.")

    # index all columns
    db_path = duckdb_index_file_directory.resolve() / index_filename
    con = duckdb.connect(str(db_path.resolve()))

    hf_api = HfApi(endpoint=hf_endpoint, token=hf_token)
    stemmer = None

    try:
        if transformed_df is not None:
            logging.debug(transformed_df.head())
            # update original data with results of transformations (string lengths, audio durations, etc.):
            logging.info(f"Updating data with {transformed_df.columns}")
            create_command_sql = CREATE_TABLE_JOIN_WITH_TRANSFORMED_DATA_COMMAND.format(
                columns=column_names, source=all_split_parquets
            )

        else:
            create_command_sql = CREATE_TABLE_COMMAND.format(columns=column_names, source=all_split_parquets)

        logging.info(create_command_sql)
        con.sql(create_command_sql)
        con.sql(CREATE_INDEX_ID_COLUMN_COMMANDS)
        logging.debug(con.sql("SELECT * FROM data LIMIT 5;"))
        logging.debug(con.sql("SELECT count(*) FROM data;"))

        if len(indexable_columns) > 0:
            # configure duckdb extensions
            if extensions_directory is not None:
                con.execute(SET_EXTENSIONS_DIRECTORY_COMMAND.format(directory=extensions_directory))
            con.execute(INSTALL_AND_LOAD_EXTENSION_COMMAND)
            stemmer = get_monolingual_stemmer(hf_api.dataset_info(repo_id=dataset).card_data)
            create_index_sql = CREATE_INDEX_COMMAND.format(columns=indexable_columns, stemmer=stemmer)
            logging.info(create_index_sql)
            con.sql(create_index_sql)

    finally:
        con.close()

    logging.info(f"about to push index file to {target_revision}")
    committer_hf_api = HfApi(endpoint=hf_endpoint, token=committer_hf_token)
    index_file_location = f"{config}/{split_directory}/{index_filename}"

    try:
        with lock.git_branch(
            dataset=dataset,
            branch=target_revision,
            owner=job_id,
            sleeps=LOCK_GIT_BRANCH_RETRY_SLEEPS,
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
            all_repo_files: set[str] = {f.rfilename for f in target_dataset_info.siblings}
            delete_operations = get_delete_operations(
                all_repo_files=all_repo_files,
                split_names=get_split_names(dataset=dataset, config=config),
                config=config,
            )
            logging.debug(f"delete operations for {dataset=} {delete_operations=}")

            # send the files to the target revision
            add_operations: list[CommitOperation] = [
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
    features[ROW_IDX_COLUMN] = {"dtype": "int64", "_type": "Value"}

    return SplitDuckdbIndex(
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
        partial=partial,
        num_rows=num_rows,
        num_bytes=num_bytes,
        duckdb_version=duckdb.__version__,
        stemmer=stemmer,
    )


class SplitDuckDbIndexJobRunner(SplitJobRunnerWithCache):
    duckdb_index_config: DuckDbIndexConfig

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        duckdb_index_cache_directory: StrPath,
        parquet_metadata_directory: StrPath,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            cache_directory=Path(duckdb_index_cache_directory) / DUCKDB_INDEX_JOB_RUNNER_SUBDIRECTORY,
        )
        self.duckdb_index_config = app_config.duckdb_index
        self.parquet_metadata_directory = parquet_metadata_directory

    @staticmethod
    def get_job_type() -> str:
        return "split-duckdb-index"

    def compute(self) -> CompleteJobResult:
        if self.cache_subdirectory is None:
            raise CacheDirectoryNotInitializedError("Cache directory has not been initialized.")
        return CompleteJobResult(
            compute_split_duckdb_index_response(
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
                source_revision=self.app_config.parquet_and_info.target_revision,
                max_split_size_bytes=self.duckdb_index_config.max_split_size_bytes,
                parquet_metadata_directory=self.parquet_metadata_directory,
            )
        )
