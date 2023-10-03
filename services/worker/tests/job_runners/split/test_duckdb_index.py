# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import os
from collections.abc import Callable
from dataclasses import replace
from http import HTTPStatus
from typing import Optional

import duckdb
import pandas as pd
import pytest
import requests
from datasets import Features, Image, Sequence, Value
from libcommon.config import ProcessingGraphConfig
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response
from libcommon.storage import StrPath
from libcommon.utils import Priority

from worker.config import AppConfig
from worker.job_runners.config.parquet_and_info import ConfigParquetAndInfoJobRunner
from worker.job_runners.split.duckdb_index import (
    CREATE_INDEX_COMMAND,
    CREATE_SEQUENCE_COMMAND,
    CREATE_TABLE_COMMAND,
    SplitDuckDbIndexJobRunner,
    get_indexable_columns,
)
from worker.resources import LibrariesResource

from ...fixtures.hub import HubDatasetTest

GetJobRunner = Callable[[str, str, str, AppConfig], SplitDuckDbIndexJobRunner]

GetParquetJobRunner = Callable[[str, str, AppConfig], ConfigParquetAndInfoJobRunner]


@pytest.fixture
def get_job_runner(
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
        processing_step_name = SplitDuckDbIndexJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            ProcessingGraphConfig(
                {
                    "dataset-step": {"input_type": "dataset"},
                    "config-parquet": {
                        "input_type": "config",
                        "triggered_by": "dataset-step",
                        "provides_config_parquet": True,
                    },
                    "config-split-names-from-streaming": {
                        "input_type": "config",
                        "triggered_by": "dataset-step",
                    },
                    processing_step_name: {
                        "input_type": "dataset",
                        "job_runner_version": SplitDuckDbIndexJobRunner.get_job_runner_version(),
                        "triggered_by": ["config-parquet", "config-split-names-from-streaming"],
                    },
                }
            )
        )

        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        upsert_response(
            kind="config-split-names-from-streaming",
            dataset=dataset,
            config=config,
            content={"splits": [{"dataset": dataset, "config": config, "split": split}]},
            http_status=HTTPStatus.OK,
        )

        return SplitDuckDbIndexJobRunner(
            job_info={
                "type": SplitDuckDbIndexJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
                    "config": config,
                    "split": split,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 50,
            },
            app_config=app_config,
            processing_step=processing_graph.get_processing_step(processing_step_name),
            duckdb_index_cache_directory=duckdb_index_cache_directory,
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
    ) -> ConfigParquetAndInfoJobRunner:
        processing_step_name = ConfigParquetAndInfoJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            ProcessingGraphConfig(
                {
                    "dataset-level": {"input_type": "dataset"},
                    processing_step_name: {
                        "input_type": "config",
                        "job_runner_version": ConfigParquetAndInfoJobRunner.get_job_runner_version(),
                        "triggered_by": "dataset-level",
                    },
                }
            )
        )

        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        return ConfigParquetAndInfoJobRunner(
            job_info={
                "type": ConfigParquetAndInfoJobRunner.get_job_type(),
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
            processing_step=processing_graph.get_processing_step(processing_step_name),
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "hub_dataset_name,max_parquet_size_bytes,expected_rows_count,expected_has_fts,expected_error_code",
    [
        ("duckdb_index", None, 5, True, None),
        ("partial_duckdb_index", None, 5, True, None),
        ("gated", None, 5, True, None),
        ("duckdb_index", 1_000, 5, False, "SplitWithTooBigParquetError"),  # parquet size is 2812
        ("public", None, 4, False, None),  # dataset does not have string columns to index
    ],
)
def test_compute(
    get_parquet_job_runner: GetParquetJobRunner,
    get_job_runner: GetJobRunner,
    app_config: AppConfig,
    hub_responses_public: HubDatasetTest,
    hub_responses_duckdb_index: HubDatasetTest,
    hub_responses_gated_duckdb_index: HubDatasetTest,
    hub_dataset_name: str,
    max_parquet_size_bytes: Optional[int],
    expected_has_fts: bool,
    expected_rows_count: int,
    expected_error_code: str,
) -> None:
    hub_datasets = {
        "public": hub_responses_public,
        "duckdb_index": hub_responses_duckdb_index,
        "partial_duckdb_index": hub_responses_duckdb_index,
        "gated": hub_responses_gated_duckdb_index,
    }
    dataset = hub_datasets[hub_dataset_name]["name"]
    config = hub_datasets[hub_dataset_name]["config_names_response"]["config_names"][0]["config"]
    split = "train"
    partial = hub_dataset_name.startswith("partial_")

    app_config = (
        app_config
        if max_parquet_size_bytes is None
        else replace(
            app_config, duckdb_index=replace(app_config.duckdb_index, max_parquet_size_bytes=max_parquet_size_bytes)
        )
    )
    app_config = (
        app_config
        if not partial
        else replace(
            app_config,
            parquet_and_info=replace(
                app_config.parquet_and_info, max_dataset_size=1, max_row_group_byte_size_for_copy=1
            ),
        )
    )

    parquet_job_runner = get_parquet_job_runner(dataset, config, app_config)
    parquet_response = parquet_job_runner.compute()
    config_parquet = parquet_response.content

    assert config_parquet["partial"] is partial

    # TODO: simulate more than one parquet file to index
    upsert_response(
        "config-parquet-and-info",
        dataset=dataset,
        config=config,
        http_status=HTTPStatus.OK,
        content=config_parquet,
    )

    assert parquet_response
    job_runner = get_job_runner(dataset, config, split, app_config)
    job_runner.pre_compute()

    if expected_error_code:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.typename == expected_error_code
    else:
        job_runner.pre_compute()
        response = job_runner.compute()
        assert response
        content = response.content
        url = content["url"]
        file_name = content["filename"]
        features = content["features"]
        has_fts = content["has_fts"]
        assert isinstance(has_fts, bool)
        assert has_fts == expected_has_fts
        assert isinstance(url, str)
        if partial:
            assert url.rsplit("/", 2)[1] == "partial-" + split
        else:
            assert url.rsplit("/", 2)[1] == split
        assert file_name is not None
        assert Features.from_dict(features) is not None
        job_runner.post_compute()

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
    "df, query, expected_ids",
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
    duckdb.sql(CREATE_SEQUENCE_COMMAND)
    duckdb.sql(CREATE_TABLE_COMMAND.format(columns=columns) + " df;")
    duckdb.sql(CREATE_INDEX_COMMAND.format(columns=columns))
    result = duckdb.execute(FTS_COMMAND, parameters=[query]).df()
    assert list(result.__hf_index_id) == expected_ids
