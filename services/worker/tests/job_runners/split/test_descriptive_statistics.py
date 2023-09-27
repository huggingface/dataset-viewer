# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.
import os
from collections.abc import Callable, Mapping
from http import HTTPStatus
from typing import Optional

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response
from libcommon.storage import StrPath
from libcommon.utils import Priority

from worker.config import AppConfig
from worker.job_runners.config.parquet_and_info import ConfigParquetAndInfoJobRunner
from worker.job_runners.split.descriptive_statistics import (
    DECIMALS,
    MAX_NUM_STRING_LABELS,
    ColumnType,
    SplitDescriptiveStatisticsJobRunner,
    generate_bins,
)
from worker.resources import LibrariesResource

from ...fixtures.hub import HubDatasetTest

GetJobRunner = Callable[[str, str, str, AppConfig], SplitDescriptiveStatisticsJobRunner]

GetParquetAndInfoJobRunner = Callable[[str, str, AppConfig], ConfigParquetAndInfoJobRunner]


@pytest.fixture
def get_job_runner(
    statistics_cache_directory: StrPath,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        split: str,
        app_config: AppConfig,
    ) -> SplitDescriptiveStatisticsJobRunner:
        processing_step_name = SplitDescriptiveStatisticsJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            {
                "dataset-config-names": {"input_type": "dataset"},
                "config-split-names-from-info": {
                    "input_type": "config",
                    "triggered_by": "dataset-config-names",
                },
                processing_step_name: {
                    "input_type": "split",
                    "job_runner_version": SplitDescriptiveStatisticsJobRunner.get_job_runner_version(),
                    "triggered_by": ["config-split-names-from-info"],
                },
            }
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

        return SplitDescriptiveStatisticsJobRunner(
            job_info={
                "type": SplitDescriptiveStatisticsJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
                    "config": config,
                    "split": split,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 100,
            },
            app_config=app_config,
            processing_step=processing_graph.get_processing_step(processing_step_name),
            statistics_cache_directory=statistics_cache_directory,
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
        processing_step_name = ConfigParquetAndInfoJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            {
                "dataset-config-names": {"input_type": "dataset"},
                processing_step_name: {
                    "input_type": "config",
                    "job_runner_version": ConfigParquetAndInfoJobRunner.get_job_runner_version(),
                    "triggered_by": "dataset-config-names",
                },
            }
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
                "difficulty": 100,
            },
            app_config=app_config,
            processing_step=processing_graph.get_processing_step(processing_step_name),
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
        )

    return _get_job_runner


def count_expected_statistics_for_numerical_column(
    column: pd.Series, dtype: ColumnType  # type: ignore
) -> dict:  # type: ignore
    minimum, maximum, mean, median, std = (
        column.min(),
        column.max(),
        column.mean(),
        column.median(),
        column.std(),
    )
    n_samples = column.shape[0]
    nan_count = column.isna().sum()
    if dtype is ColumnType.FLOAT:
        hist, bin_edges = np.histogram(column[~column.isna()])
        bin_edges = bin_edges.astype(float).round(DECIMALS).tolist()
    else:
        n_bins = int(os.getenv("DESCRIPTIVE_STATISTICS_HISTOGRAM_NUM_BINS", 10))
        bins = generate_bins(minimum, maximum, column_name="dummy", column_type=dtype, n_bins=n_bins)
        hist, bin_edges = np.histogram(column[~column.isna()], np.append(bins.bin_min, maximum))
        bin_edges = bin_edges.astype(int).tolist()
    hist = hist.astype(int).tolist()
    if dtype is ColumnType.FLOAT:
        minimum = minimum.astype(float).round(DECIMALS).item()
        maximum = maximum.astype(float).round(DECIMALS).item()
        mean = mean.astype(float).round(DECIMALS).item()  # type: ignore
        median = median.astype(float).round(DECIMALS).item()  # type: ignore
        std = std.astype(float).round(DECIMALS).item()  # type: ignore
    else:
        mean, median, std = list(np.round([mean, median, std], DECIMALS))
    return {
        "nan_count": nan_count,
        "nan_proportion": np.round(nan_count / n_samples, DECIMALS).item() if nan_count else 0.0,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "median": median,
        "std": std,
        "histogram": {
            "hist": hist,
            "bin_edges": bin_edges,
        },
    }


def count_expected_statistics_for_categorical_column(
    column: pd.Series, class_labels: list[str]  # type: ignore
) -> dict:  # type: ignore
    n_samples = column.shape[0]
    nan_count = column.isna().sum()
    value_counts = column.value_counts().to_dict()
    n_unique = len(value_counts)
    frequencies = {class_labels[int(class_id)]: class_count for class_id, class_count in value_counts.items()}
    return {
        "nan_count": nan_count,
        "nan_proportion": np.round(nan_count / n_samples, DECIMALS).item() if nan_count else 0.0,
        "n_unique": n_unique,
        "frequencies": frequencies,
    }


def count_expected_statistics_for_string_column(column: pd.Series) -> dict:  # type: ignore
    n_samples = column.shape[0]
    nan_count = column.isna().sum()
    value_counts = column.value_counts().to_dict()
    n_unique = len(value_counts)
    if n_unique <= MAX_NUM_STRING_LABELS:
        return {
            "nan_count": nan_count,
            "nan_proportion": np.round(nan_count / n_samples, DECIMALS).item() if nan_count else 0.0,
            "n_unique": n_unique,
            "frequencies": value_counts,
        }

    lengths_column = column.map(lambda x: len(x) if x is not None else None)
    return count_expected_statistics_for_numerical_column(lengths_column, dtype=ColumnType.INT)


@pytest.fixture
def descriptive_statistics_expected(datasets: Mapping[str, Dataset]) -> dict:  # type: ignore
    ds = datasets["descriptive_statistics"]
    df = ds.to_pandas()
    expected_statistics = {}
    for column_name in df.columns:
        column_type = ColumnType(column_name.split("__")[0])
        if column_type is ColumnType.STRING_LABEL:
            column_stats = count_expected_statistics_for_string_column(df[column_name])
        elif column_type in [ColumnType.FLOAT, ColumnType.INT]:
            column_stats = count_expected_statistics_for_numerical_column(df[column_name], dtype=column_type)
            if sum(column_stats["histogram"]["hist"]) != df.shape[0] - column_stats["nan_count"]:
                raise ValueError(column_name, column_stats)
        elif column_type is ColumnType.CLASS_LABEL:
            class_labels = ds.features[column_name].names
            column_stats = count_expected_statistics_for_categorical_column(df[column_name], class_labels=class_labels)
        expected_statistics[column_name] = {
            "column_name": column_name,
            "column_type": column_type,
            "column_statistics": column_stats,
        }
    return {"num_examples": df.shape[0], "statistics": expected_statistics}


@pytest.fixture
def descriptive_statistics_string_text_expected(datasets: Mapping[str, Dataset]) -> dict:  # type: ignore
    ds = datasets["descriptive_statistics_string_text"]
    df = ds.to_pandas()
    expected_statistics = {}
    for column_name in df.columns:
        column_stats = count_expected_statistics_for_string_column(df[column_name])
        if sum(column_stats["histogram"]["hist"]) != df.shape[0] - column_stats["nan_count"]:
            raise ValueError(column_name, column_stats)
        expected_statistics[column_name] = {
            "column_name": column_name,
            "column_type": ColumnType.STRING_TEXT,
            "column_statistics": column_stats,
        }
    return {"num_examples": df.shape[0], "statistics": expected_statistics}


@pytest.mark.parametrize(
    "hub_dataset_name,expected_error_code",
    [
        ("descriptive_statistics", None),
        ("descriptive_statistics_string_text", None),
        ("gated", None),
        ("audio", "NoSupportedFeaturesError"),
        ("big", "SplitWithTooBigParquetError"),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    get_parquet_and_info_job_runner: GetParquetAndInfoJobRunner,
    hub_responses_descriptive_statistics: HubDatasetTest,
    hub_responses_descriptive_statistics_string_text: HubDatasetTest,
    hub_responses_gated_descriptive_statistics: HubDatasetTest,
    hub_responses_audio: HubDatasetTest,
    hub_responses_big: HubDatasetTest,
    hub_dataset_name: str,
    expected_error_code: Optional[str],
    descriptive_statistics_expected: dict,  # type: ignore
    descriptive_statistics_string_text_expected: dict,  # type: ignore
) -> None:
    hub_datasets = {
        "descriptive_statistics": hub_responses_descriptive_statistics,
        "descriptive_statistics_string_text": hub_responses_descriptive_statistics_string_text,
        "gated": hub_responses_gated_descriptive_statistics,
        "audio": hub_responses_audio,
        "big": hub_responses_big,
    }
    expected = {
        "descriptive_statistics": descriptive_statistics_expected,
        "gated": descriptive_statistics_expected,
        "descriptive_statistics_string_text": descriptive_statistics_string_text_expected,
    }
    dataset = hub_datasets[hub_dataset_name]["name"]
    splits_response = hub_datasets[hub_dataset_name]["splits_response"]
    config, split = splits_response["splits"][0]["config"], splits_response["splits"][0]["split"]
    expected_response = expected.get(hub_dataset_name)

    # computing and pushing real parquet files because we need them for stats computation
    parquet_job_runner = get_parquet_and_info_job_runner(dataset, config, app_config)
    parquet_and_info_response = parquet_job_runner.compute()

    upsert_response(
        "config-parquet-and-info",
        dataset=dataset,
        config=config,
        http_status=HTTPStatus.OK,
        content=parquet_and_info_response.content,
    )

    assert parquet_and_info_response
    job_runner = get_job_runner(dataset, config, split, app_config)
    job_runner.pre_compute()
    if expected_error_code:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.typename == expected_error_code
    else:
        response = job_runner.compute()
        assert sorted(response.content.keys()) == ["num_examples", "statistics"]
        assert response.content["num_examples"] == expected_response["num_examples"]  # type: ignore
        response_statistics = response.content["statistics"]
        expected_statistics = expected_response["statistics"]  # type: ignore
        assert len(response_statistics) == len(expected_statistics)
        assert set([column_response["column_name"] for column_response in response_statistics]) == set(
            expected_statistics
        )  # assert returned features are as expected
        for column_response_statistics in response_statistics:
            assert_statistics_equal(
                column_response_statistics, expected_statistics[column_response_statistics["column_name"]]
            )


def assert_statistics_equal(response: dict, expected: dict) -> None:  # type: ignore
    """
    Check that all values are equal or in case of float - almost equal.
    We use np.isclose because of small possible mismatches
    between numpy (which is used for counting expected values) and python float rounding.
    """
    assert response["column_name"] == expected["column_name"]
    assert response["column_type"] == expected["column_type"]
    response_stats, expected_stats = response["column_statistics"], expected["column_statistics"]
    assert response_stats.keys() == expected_stats.keys()

    assert response_stats["nan_count"] == expected_stats["nan_count"]
    assert np.isclose(response_stats["nan_proportion"], expected_stats["nan_proportion"], 1e-3)

    if response["column_type"] is ColumnType.FLOAT:
        assert np.isclose(
            response_stats["histogram"]["bin_edges"], expected_stats["histogram"]["bin_edges"], 1e-3
        ).all()
        assert np.isclose(response_stats["min"], expected_stats["min"], 1e-3)
        assert np.isclose(response_stats["max"], expected_stats["max"], 1e-3)
        assert np.isclose(response_stats["mean"], expected_stats["mean"], 1e-3)
        assert np.isclose(response_stats["median"], expected_stats["median"], 1e-3)
        assert np.isclose(response_stats["std"], expected_stats["std"], 1e-3)
        assert response_stats["histogram"]["hist"] == expected_stats["histogram"]["hist"]

    elif response["column_type"] is ColumnType.INT:
        assert np.isclose(response_stats["mean"], expected_stats["mean"], 1e-3)
        assert np.isclose(response_stats["median"], expected_stats["median"], 1e-3)
        assert np.isclose(response_stats["std"], expected_stats["std"], 1e-3)
        assert response_stats["min"] == expected_stats["min"]
        assert response_stats["max"] == expected_stats["max"]
        assert response_stats["histogram"] == expected_stats["histogram"]

    elif response["column_type"] is ColumnType.CLASS_LABEL:
        assert response_stats["n_unique"] == expected_stats["n_unique"]
        assert response_stats["frequencies"] == expected_stats["frequencies"]

    elif response["column_type"] is ColumnType.STRING_LABEL:
        assert response_stats["n_unique"] == expected_stats["n_unique"]
        assert response_stats["frequencies"] == expected_stats["frequencies"]

    elif response["column_type"] is ColumnType.STRING_TEXT:
        assert np.isclose(response_stats["mean"], expected_stats["mean"], 1e-3)
        assert np.isclose(response_stats["median"], expected_stats["median"], 1e-3)
        assert np.isclose(response_stats["std"], expected_stats["std"], 1e-3)
        assert response_stats["min"] == expected_stats["min"]
        assert response_stats["max"] == expected_stats["max"]
        assert response_stats["histogram"] == expected_stats["histogram"]

    else:
        raise ValueError("Incorrect data type")
