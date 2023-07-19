# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from http import HTTPStatus
from typing import Callable, List, Mapping, Optional

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


EXPECTED_STATS_CONTENT = {
    "num_examples": 20,
    "stats": [
        {
            "column_name": "class_label_column",
            "column_type": "class_label",
            "column_dtype": None,
            "column_statistics": {
                "nan_count": 0,
                "nan_proportion": 0.0,
                "n_unique": 2,
                "frequencies": {"cat": 17, "dog": 3},
            },
        },
        {
            "column_name": "class_label_nan_column",
            "column_type": "class_label",
            "column_dtype": None,
            "column_statistics": {
                "nan_count": 4,
                "nan_proportion": 0.2,
                "n_unique": 3,
                "frequencies": {"cat": 15, "dog": 1},
            },
        },
        {
            "column_name": "int_column",
            "column_type": "int",
            "column_dtype": "int32",
            "column_statistics": {
                "nan_count": 0,
                "nan_proportion": 0.0,
                "min": 0,
                "max": 8,
                "mean": 4.05,
                "median": 4.5,
                "std": 2.60516,
                "histogram": {"hist": [2, 2, 3, 1, 2, 5, 1, 1, 3, 0], "bin_edges": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
            },
        },
        {
            "column_name": "int_nan_column",
            "column_type": "int",
            "column_dtype": "int32",
            "column_statistics": {
                "nan_count": 6,
                "nan_proportion": 0.3,
                "min": 0,
                "max": 8,
                "mean": 4.71429,
                "median": 5.0,
                "std": 2.64367,
                "histogram": {"hist": [1, 1, 2, 0, 1, 4, 1, 1, 3, 0], "bin_edges": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
            },
        },
        {
            "column_name": "float_column",
            "column_type": "float",
            "column_dtype": "float32",
            "column_statistics": {
                "nan_count": 0,
                "nan_proportion": 0.0,
                "min": 0.1,
                "max": 9.9,
                "mean": 4.585,
                "median": 4.9,
                "std": 3.56197,
                "histogram": {
                    "hist": [5, 1, 3, 0, 1, 1, 3, 1, 2, 3],
                    "bin_edges": [0.1, 1.08, 2.06, 3.04, 4.02, 5.0, 5.98, 6.96, 7.94, 8.92],
                },
            },
        },
        {
            "column_name": "float_nan_column",
            "column_type": "float",
            "column_dtype": "float32",
            "column_statistics": {
                "nan_count": 8,
                "nan_proportion": 0.4,
                "min": 0.2,
                "max": 9.9,
                "mean": 5.09167,
                "median": 4.9,
                "std": 3.87497,
                "histogram": {
                    "hist": [3, 0, 2, 0, 1, 1, 0, 0, 2, 3],
                    "bin_edges": [0.2, 1.17, 2.14, 3.11, 4.08, 5.05, 6.02, 6.99, 7.96, 8.93],
                },
            },
        },
        {
            "column_name": "float_negative_column",
            "column_type": "float",
            "column_dtype": "float32",
            "column_statistics": {
                "nan_count": 0,
                "nan_proportion": 0.0,
                "min": -15.754,
                "max": -5.333,
                "mean": -9.8159,
                "median": -9.0,
                "std": 3.03655,
                "histogram": {
                    "hist": [2, 1, 2, 1, 1, 2, 2, 3, 5, 1],
                    "bin_edges": [0.2, 1.17, 2.14, 3.11, 4.08, 5.05, 6.02, 6.99, 7.96, 8.93],
                },
            },
        },
    ],
}


def count_expected_statistics_for_numerical_column(column: pd.Series, dtype: str) -> dict:  # type: ignore
    minimum, maximum, mean, median, std = (
        column.min(),  # .astype(float).round(DECIMALS).item(),
        column.max(),  # .astype(float).round(DECIMALS).item(),
        column.mean(),  # .astype(float).round(DECIMALS).item(),
        column.median(),  # .astype(float).round(DECIMALS).item(),
        column.std(),  # .astype(float).round(DECIMALS).item(),
    )
    n_samples = column.shape[0]
    nan_count = column.isna().sum()
    if dtype == "FLOAT":
        hist, bin_edges = np.histogram(column[~column.isna()])
        bin_edges = bin_edges.astype(float).round(DECIMALS).tolist()
    else:
        bins = generate_bins(minimum, maximum, dtype, 10)  # TODO: N BINS
        hist, bin_edges = np.histogram(column[~column.isna()], np.append(bins.bin_min, maximum))
        bin_edges = bin_edges.astype(int).tolist()
    hist = hist.astype(int).tolist()
    if dtype == "FLOAT":
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
    column: pd.Series, class_labels: List[str]  # type: ignore
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


@pytest.fixture
def descriptive_statistics_expected(datasets: Mapping[str, Dataset]) -> dict:  # type: ignore
    columns_to_types = {
        "int_column": "INT",
        "int_nan_column": "INT",
        "float_column": "FLOAT",
        "float_nan_column": "FLOAT",
        "class_label_column": "CLASS_LABEL",
        "class_label_nan_column": "CLASS_LABEL",
        "float_negative_column": "FLOAT",
        "float_cross_zero_column": "FLOAT",
        "int_negative_column": "INT",
        "int_cross_zero_column": "INT",
    }
    ds = datasets["descriptive_statistics"]
    df = ds.to_pandas()
    expected_statistics = {}
    for column_name in df.columns:
        column_type = columns_to_types[column_name]
        if column_type in ["FLOAT", "INT"]:
            column_stats = count_expected_statistics_for_numerical_column(df[column_name], dtype=column_type)
            if sum(column_stats["histogram"]["hist"]) != df.shape[0] - column_stats["nan_count"]:
                raise ValueError(column_name, column_stats)
            expected_statistics[column_name] = {
                "column_name": column_name,
                "column_type": column_type,
                "column_statistics": column_stats,
            }
        elif column_type == "CLASS_LABEL":
            class_labels = ds.features[column_name].names
            column_stats = count_expected_statistics_for_categorical_column(df[column_name], class_labels=class_labels)
            expected_statistics[column_name] = {
                "column_name": column_name,
                "column_type": column_type,
                "column_statistics": column_stats,
            }
    return expected_statistics


@pytest.mark.parametrize(
    "hub_dataset_name,expected_error_code",
    [
        ("descriptive_statistics", None),
        ("audio", "NoSupportedFeaturesError"),
        ("big", "SplitWithTooBigParquetError"),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    get_parquet_and_info_job_runner: GetParquetAndInfoJobRunner,
    hub_responses_descriptive_statistics: HubDatasetTest,
    hub_responses_audio: HubDatasetTest,
    hub_responses_big: HubDatasetTest,
    hub_dataset_name: str,
    expected_error_code: Optional[str],
    descriptive_statistics_expected: dict,  # type: ignore
) -> None:
    hub_datasets = {
        "descriptive_statistics": hub_responses_descriptive_statistics,
        "audio": hub_responses_audio,
        "big": hub_responses_big,
    }
    dataset = hub_datasets[hub_dataset_name]["name"]
    config_names_response = hub_datasets[hub_dataset_name]["config_names_response"]
    splits_response = hub_datasets[hub_dataset_name]["splits_response"]
    config, split = splits_response["splits"][0]["config"], splits_response["splits"][0]["split"]

    upsert_response("dataset-config-names", dataset=dataset, http_status=HTTPStatus.OK, content=config_names_response)
    upsert_response(
        "config-split-names-from-info",
        dataset=dataset,
        config=config,
        http_status=HTTPStatus.OK,
        content=splits_response,
    )

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
        assert response.content["num_examples"] == 20
        response_statistics = response.content["statistics"]
        assert len(response_statistics) == len(descriptive_statistics_expected)
        assert set([column_response["column_name"] for column_response in response_statistics]) == set(
            descriptive_statistics_expected.keys()
        )  # columns
        for column_response in response_statistics:
            assert column_response == descriptive_statistics_expected[column_response["column_name"]]
