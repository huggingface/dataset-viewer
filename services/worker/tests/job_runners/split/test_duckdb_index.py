# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import os
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import replace
from http import HTTPStatus
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import datasets.config
import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests
from datasets import Dataset, Features, Image, Sequence, Value, Audio
from datasets.table import embed_table_storage
from datasets.packaged_modules.csv.csv import CsvConfig
from libcommon.dtos import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response
from libcommon.storage import StrPath

from worker.config import AppConfig
from worker.job_runners.config.parquet import ConfigParquetJobRunner
from worker.job_runners.config.parquet_and_info import ConfigParquetAndInfoJobRunner
from worker.job_runners.config.parquet_metadata import ConfigParquetMetadataJobRunner
from worker.job_runners.split.duckdb_index import (
    CREATE_INDEX_COMMAND,
    CREATE_INDEX_ID_COLUMN_COMMANDS,
    CREATE_TABLE_COMMAND,
    SplitDuckDbIndexJobRunner,
    get_delete_operations,
    get_indexable_columns,
)
from worker.resources import LibrariesResource

from ...fixtures.hub import HubDatasetTest
from ..utils import REVISION_NAME

GetJobRunner = Callable[[str, str, str, AppConfig], SplitDuckDbIndexJobRunner]

GetParquetAndInfoJobRunner = Callable[[str, str, AppConfig], ConfigParquetAndInfoJobRunner]
GetParquetJobRunner = Callable[[str, str, AppConfig], ConfigParquetJobRunner]
GetParquetMetadataJobRunner = Callable[[str, str, AppConfig], ConfigParquetMetadataJobRunner]


@pytest.fixture
def get_job_runner(
    parquet_metadata_directory: StrPath,
    duckdb_index_cache_directory: StrPath,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        split: str,
        app_config: AppConfig,
    ) -> SplitDuckDbIndexJobRunner:
        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        upsert_response(
            kind="config-split-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            config=config,
            content={"splits": [{"dataset": dataset, "config": config, "split": split}]},
            http_status=HTTPStatus.OK,
        )

        return SplitDuckDbIndexJobRunner(
            job_info={
                "type": SplitDuckDbIndexJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": REVISION_NAME,
                    "config": config,
                    "split": split,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 50,
            },
            app_config=app_config,
            duckdb_index_cache_directory=duckdb_index_cache_directory,
            parquet_metadata_directory=parquet_metadata_directory,
        )

    return _get_job_runner


@pytest.fixture
def get_parquet_and_info_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetParquetAndInfoJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
    ) -> ConfigParquetAndInfoJobRunner:
        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        return ConfigParquetAndInfoJobRunner(
            job_info={
                "type": ConfigParquetAndInfoJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": REVISION_NAME,
                    "config": config,
                    "split": None,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 50,
            },
            app_config=app_config,
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
        )

    return _get_job_runner


@pytest.fixture
def get_parquet_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetParquetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
    ) -> ConfigParquetJobRunner:
        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        return ConfigParquetJobRunner(
            job_info={
                "type": ConfigParquetJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
                    "config": config,
                    "split": None,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 50,
            },
            app_config=app_config,
        )

    return _get_job_runner


@pytest.fixture
def get_parquet_metadata_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
    parquet_metadata_directory: StrPath,
) -> GetParquetMetadataJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
    ) -> ConfigParquetMetadataJobRunner:
        upsert_response(
            kind="dataset-config-names",
            dataset_git_revision=REVISION_NAME,
            dataset=dataset,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        return ConfigParquetMetadataJobRunner(
            job_info={
                "type": ConfigParquetMetadataJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
                    "config": config,
                    "split": None,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 50,
            },
            app_config=app_config,
            parquet_metadata_directory=parquet_metadata_directory,
        )

    return _get_job_runner


expected_columns_duckdb_index = [
    "text",
    "column with spaces",
    "list",
    "sequence_list",
    "sequence_struct",
    "audio",
    "image",
    "text__hf_len",
    "column with spaces__hf_len",
    "list__hf_len",
    "sequence_list__hf_len",
    "audio__hf_duration",
    "image__hf_width",
    "image__hf_height",
    "__hf_index_id",
]
expected_columns_multiple_files = [  # no text or transformable columns in `hub_public_csv` dataset
    "col_1",
    "col_2",
    "col_3",  # note that there is no `__hf_index_id` column because there is no search index
]


@pytest.fixture
def expected_values(datasets: Mapping[str, Dataset]) -> dict[str, list[Any]]:
    ds = datasets["duckdb_index"]
    ds = Dataset(embed_table_storage(ds.data))
    expected: dict[str, list[Any]] = ds[:]
    for feature_name, feature in ds.features.items():
        is_string = isinstance(feature, Value) and feature.dtype == "string"
        is_list = (isinstance(feature, list) or isinstance(feature, Sequence)) and feature_name != "sequence_struct"
        if is_string or is_list:
            expected[f"{feature_name}__hf_len"] = [len(row) if row is not None else None for row in ds[feature_name]]
        elif isinstance(feature, Audio):
            expected[f"{feature_name}__hf_duration"] = [1.0, 2.0, 3.0, 4.0, None]
        elif isinstance(feature, Image):
            expected[f"{feature_name}__hf_width"] = [640, 1440, 520, 1240, None]
            expected[f"{feature_name}__hf_height"] = [480, 1058, 400, 930, None]
    expected["__hf_index_id"] = list(range(ds.num_rows))
    return expected


@pytest.mark.parametrize(
    "hub_dataset_name,max_split_size_bytes,expected_rows_count,expected_has_fts,expected_partial,expected_error_code,expected_columns",
    [
        ("duckdb_index", None, 5, True, False, None, expected_columns_duckdb_index),
        ("duckdb_index_from_partial_export", None, 5, True, True, None, expected_columns_duckdb_index),
        ("gated", None, 5, True, False, None, expected_columns_duckdb_index),
        ("partial_duckdb_index_from_multiple_files_public", 1, 1, False, True, None, expected_columns_multiple_files),
    ],
)
def test_compute(
    get_parquet_and_info_job_runner: GetParquetAndInfoJobRunner,
    get_parquet_job_runner: GetParquetJobRunner,
    get_parquet_metadata_job_runner: GetParquetMetadataJobRunner,
    get_job_runner: GetJobRunner,
    app_config: AppConfig,
    hub_responses_public: HubDatasetTest,
    hub_responses_duckdb_index: HubDatasetTest,
    hub_responses_gated_duckdb_index: HubDatasetTest,
    hub_dataset_name: str,
    max_split_size_bytes: Optional[int],
    expected_has_fts: bool,
    expected_rows_count: int,
    expected_partial: bool,
    expected_error_code: str,
    expected_columns: Optional[list[str]],
    expected_values: dict[str, list[Any]],
) -> None:
    hub_datasets = {
        "duckdb_index": hub_responses_duckdb_index,
        "duckdb_index_from_partial_export": hub_responses_duckdb_index,
        "gated": hub_responses_gated_duckdb_index,
        "partial_duckdb_index_from_multiple_files_public": hub_responses_public,
    }
    dataset = hub_datasets[hub_dataset_name]["name"]
    config = hub_datasets[hub_dataset_name]["config_names_response"]["config_names"][0]["config"]
    split = "train"
    partial_parquet_export = hub_dataset_name == "duckdb_index_from_partial_export"
    multiple_parquet_files = hub_dataset_name == "partial_duckdb_index_from_multiple_files_public"

    app_config = (
        app_config
        if max_split_size_bytes is None
        else replace(
            app_config, duckdb_index=replace(app_config.duckdb_index, max_split_size_bytes=max_split_size_bytes)
        )
    )
    app_config = (
        app_config
        if not partial_parquet_export
        else replace(
            app_config,
            parquet_and_info=replace(
                app_config.parquet_and_info, max_dataset_size_bytes=1, max_row_group_byte_size_for_copy=1
            ),
        )
    )

    parquet_and_info_job_runner = get_parquet_and_info_job_runner(dataset, config, app_config)
    with ExitStack() as stack:
        if multiple_parquet_files:
            stack.enter_context(patch.object(datasets.config, "MAX_SHARD_SIZE", 1))
            # Set a small chunk size to yield more than one Arrow Table in _generate_tables
            # to be able to generate multiple tables and therefore multiple files
            stack.enter_context(patch.object(CsvConfig, "pd_read_csv_kwargs", {"chunksize": 1}))
        parquet_and_info_response = parquet_and_info_job_runner.compute()
    config_parquet_and_info = parquet_and_info_response.content
    if multiple_parquet_files:
        assert len(config_parquet_and_info["parquet_files"]) > 1

    assert config_parquet_and_info["partial"] is partial_parquet_export

    upsert_response(
        "config-parquet-and-info",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        http_status=HTTPStatus.OK,
        content=config_parquet_and_info,
    )

    parquet_job_runner = get_parquet_job_runner(dataset, config, app_config)
    parquet_response = parquet_job_runner.compute()
    config_parquet = parquet_response.content

    assert config_parquet["partial"] is partial_parquet_export

    upsert_response(
        "config-parquet",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        http_status=HTTPStatus.OK,
        content=config_parquet,
    )

    parquet_metadata_job_runner = get_parquet_metadata_job_runner(dataset, config, app_config)
    parquet_metadata_response = parquet_metadata_job_runner.compute()
    config_parquet_metadata = parquet_metadata_response.content
    parquet_export_num_rows = sum(
        parquet_file["num_rows"] for parquet_file in config_parquet_metadata["parquet_files_metadata"]
    )

    assert config_parquet_metadata["partial"] is partial_parquet_export

    upsert_response(
        "config-parquet-metadata",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        http_status=HTTPStatus.OK,
        content=config_parquet_metadata,
    )

    # setup is ready, test starts here
    job_runner = get_job_runner(dataset, config, split, app_config)
    job_runner.pre_compute()

    if expected_error_code:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.typename == expected_error_code
    else:
        with patch("worker.job_runners.split.duckdb_index.StringColumn.is_class") as mock:
            mock.return_value = False
            response = job_runner.compute()
        assert response
        content = response.content
        url = content["url"]
        file_name = content["filename"]
        features = content["features"]
        has_fts = content["has_fts"]
        partial = content["partial"]
        assert isinstance(has_fts, bool)
        assert has_fts == expected_has_fts
        assert isinstance(url, str)
        if partial_parquet_export:
            assert url.rsplit("/", 2)[1] == "partial-" + split
        else:
            assert url.rsplit("/", 2)[1] == split
        assert file_name is not None
        assert Features.from_dict(features) is not None
        assert isinstance(partial, bool)
        assert partial == expected_partial
        if content["num_rows"] < parquet_export_num_rows:
            assert url.rsplit("/", 1)[1] == "partial-index.duckdb"
        else:
            assert url.rsplit("/", 1)[1] == "index.duckdb"

        # download locally duckdb index file
        duckdb_file = requests.get(url, headers={"authorization": f"Bearer {app_config.common.hf_token}"})
        with open(file_name, "wb") as f:
            f.write(duckdb_file.content)

        duckdb.execute("INSTALL 'fts';")
        duckdb.execute("LOAD 'fts';")
        con = duckdb.connect(file_name)

        # validate number of inserted records
        record_count = con.sql("SELECT COUNT(*) FROM data;").fetchall()
        assert record_count is not None
        assert isinstance(record_count, list)
        assert record_count[0] == (expected_rows_count,)

        columns = [row[0] for row in con.sql("SELECT column_name FROM (DESCRIBE data);").fetchall()]
        assert columns == expected_columns
        if not multiple_parquet_files:
            data = con.sql("SELECT * FROM data;").fetchall()
            data = {column_name: list(values) for column_name, values in zip(columns, zip(*data))}
            assert data == expected_values

        if has_fts:
            # perform a search to validate fts feature
            query = "Lord Vader"
            result = con.execute(
                "SELECT __hf_index_id, text FROM data WHERE fts_main_data.match_bm25(__hf_index_id, ?) IS NOT NULL;",
                [query],
            )
            rows = result.df()
            assert rows is not None
            assert (rows["text"].eq("Vader turns round and round in circles as his ship spins into space.")).any()
            assert (rows["text"].eq("The wingman spots the pirateship coming at him and warns the Dark Lord")).any()
            assert (rows["text"].eq("We count thirty Rebel ships, Lord Vader.")).any()
            assert (
                rows["text"].eq(
                    "Grand Moff Tarkin and Lord Vader are interrupted in their discussion by the buzz of the comlink"
                )
            ).any()
            assert not (rows["text"].eq("There goes another one.")).any()
            assert (rows["__hf_index_id"].isin([0, 2, 3, 4, 5, 7, 8, 9])).all()

        con.close()
        os.remove(file_name)
    job_runner.post_compute()


@pytest.mark.parametrize(
    "split_names,config,deleted_files",
    [
        (
            {"s1", "s2", "s3"},
            "c1",
            set(),
        ),
        (
            {"s1", "s3"},
            "c2",
            {"c2/s2/index.duckdb"},
        ),
        (
            {"s1"},
            "c2",
            {"c2/s2/index.duckdb", "c2/partial-s3/partial-index.duckdb"},
        ),
    ],
)
def test_get_delete_operations(split_names: set[str], config: str, deleted_files: set[str]) -> None:
    all_repo_files = {
        "c1/s1/000.parquet",
        "c1/s1/index.duckdb",
        "c2/s1/000.parquet",
        "c2/s1/index.duckdb",
        "c2/s2/000.parquet",
        "c2/s2/index.duckdb",
        "c2/partial-s3/000.parquet",
        "c2/partial-s3/partial-index.duckdb",
    }
    delete_operations = get_delete_operations(all_repo_files=all_repo_files, split_names=split_names, config=config)
    assert set(delete_operation.path_in_repo for delete_operation in delete_operations) == deleted_files


@pytest.mark.parametrize(
    "features, expected",
    [
        (Features({"col_1": Value("string"), "col_2": Value("int64")}), ["col_1"]),
        (
            Features(
                {
                    "nested_1": [Value("string")],
                    "nested_2": Sequence(Value("string")),
                    "nested_3": Sequence({"foo": Value("string")}),
                    "nested_4": {"foo": Value("string"), "bar": Value("int64")},
                    "nested_int": [Value("int64")],
                }
            ),
            ["nested_1", "nested_2", "nested_3", "nested_4"],
        ),
        (Features({"col_1": Image()}), []),
    ],
)
def test_get_indexable_columns(features: Features, expected: list[str]) -> None:
    indexable_columns = get_indexable_columns(features)
    assert indexable_columns == expected


DATA = """Hello there !
General Kenobi.
You are a bold one.
Kill him !
...
Back away ! I will deal with this Jedi slime myself"""


FTS_COMMAND = (
    "SELECT * EXCLUDE (__hf_fts_score) FROM (SELECT *, fts_main_data.match_bm25(__hf_index_id, ?) AS __hf_fts_score"
    " FROM data) A WHERE __hf_fts_score IS NOT NULL ORDER BY __hf_index_id;"
)


@pytest.mark.parametrize(
    "df,query,expected_ids",
    [
        (pd.DataFrame([{"line": line} for line in DATA.split("\n")]), "bold", [2]),
        (pd.DataFrame([{"nested": [line]} for line in DATA.split("\n")]), "bold", [2]),
        (pd.DataFrame([{"nested": {"foo": line}} for line in DATA.split("\n")]), "bold", [2]),
        (pd.DataFrame([{"nested": [{"foo": line}]} for line in DATA.split("\n")]), "bold", [2]),
        (pd.DataFrame([{"nested": [{"foo": line, "bar": 0}]} for line in DATA.split("\n")]), "bold", [2]),
    ],
)
def test_index_command(df: pd.DataFrame, query: str, expected_ids: list[int]) -> None:
    columns = ",".join('"' + str(column) + '"' for column in df.columns)
    duckdb.sql(CREATE_TABLE_COMMAND.format(columns=columns, source="df"))
    duckdb.sql(CREATE_INDEX_ID_COLUMN_COMMANDS)
    duckdb.sql(CREATE_INDEX_COMMAND.format(columns=columns))
    result = duckdb.execute(FTS_COMMAND, parameters=[query]).df()
    assert list(result.__hf_index_id) == expected_ids


def test_table_column_hf_index_id_is_monotonic_increasing(tmp_path: Path) -> None:
    pa_table = pa.Table.from_pydict({"text": [f"text-{i}" for i in range(100)]})
    parquet_path = str(tmp_path / "0000.parquet")
    pq.write_table(pa_table, parquet_path, row_group_size=2)
    db_path = str(tmp_path / "index.duckdb")
    column_names = ",".join(f'"{column}"' for column in pa_table.column_names)
    with duckdb.connect(db_path) as con:
        con.sql(CREATE_TABLE_COMMAND.format(columns=column_names, source=parquet_path))
        con.sql(CREATE_INDEX_ID_COLUMN_COMMANDS)
    with duckdb.connect(db_path) as con:
        df = con.sql("SELECT * FROM data").to_df()
    assert df["__hf_index_id"].is_monotonic_increasing
    assert df["__hf_index_id"].is_unique
