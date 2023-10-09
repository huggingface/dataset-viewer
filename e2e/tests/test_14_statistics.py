# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from .utils import get_default_config_split, poll_until_ready_and_assert


def test_statistics_endpoint(hf_public_dataset_repo_csv_data: str) -> None:
    # TODO: add dataset with various splits, or various configs
    dataset = hf_public_dataset_repo_csv_data
    config, split = get_default_config_split()
    statistics_response = poll_until_ready_and_assert(
        relative_url=f"/statistics?dataset={dataset}&config={config}&split={split}",
        check_x_revision=True,
    )

    content = statistics_response.json()
    assert "num_examples" in content, statistics_response
    assert "statistics" in content, statistics_response
    statistics = content["statistics"]
    num_examples = content["num_examples"]

    assert isinstance(statistics, list), statistics
    assert len(statistics) == 4
    assert num_examples == 4

    first_column = statistics[0]
    assert "column_name" in first_column
    assert "column_statistics" in first_column
    assert "column_type" in first_column
    assert first_column["column_name"] == "col_1"
    assert first_column["column_type"] == "string_label"
    assert isinstance(first_column["column_statistics"], dict)
    assert first_column["column_statistics"] == {
        "nan_count": 0,
        "nan_proportion": 0.0,
        "no_label_count": 0,
        "no_label_proportion": 0.0,
        "n_unique": 4,
        "frequencies": {
            "There goes another one.": 1,
            "Vader turns round and round in circles as his ship spins into space.": 1,
            "We count thirty Rebel ships, Lord Vader.": 1,
            "The wingman spots the pirateship coming at him and warns the Dark Lord": 1,
        },
    }

    second_column = statistics[1]
    assert "column_name" in second_column
    assert "column_statistics" in second_column
    assert "column_type" in second_column
    assert second_column["column_name"] == "col_2"
    assert second_column["column_type"] == "int"
    assert isinstance(second_column["column_statistics"], dict)
    assert second_column["column_statistics"] == {
        "histogram": {"bin_edges": [0, 1, 2, 3, 3], "hist": [1, 1, 1, 1]},
        "max": 3,
        "mean": 1.5,
        "median": 1.5,
        "min": 0,
        "nan_count": 0,
        "nan_proportion": 0.0,
        "std": 1.29099,
    }

    third_column = statistics[2]
    assert "column_name" in third_column
    assert "column_statistics" in third_column
    assert "column_type" in third_column
    assert third_column["column_name"] == "col_3"
    assert third_column["column_type"] == "float"
    assert isinstance(third_column["column_statistics"], dict)
    assert third_column["column_statistics"] == {
        "nan_count": 0,
        "nan_proportion": 0.0,
        "min": 0.0,
        "max": 3.0,
        "mean": 1.5,
        "median": 1.5,
        "std": 1.29099,
        "histogram": {
            "hist": [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            "bin_edges": [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0],
        },
    }

    fourth_column = statistics[3]
    assert "column_name" in fourth_column
    assert "column_statistics" in fourth_column
    assert "column_type" in fourth_column
    assert fourth_column["column_name"] == "col_4"
    assert fourth_column["column_type"] == "string_label"
    assert isinstance(fourth_column["column_statistics"], dict)
    assert fourth_column["column_statistics"] == {
        "nan_count": 0,
        "nan_proportion": 0.0,
        "no_label_count": 0,
        "no_label_proportion": 0.0,
        "n_unique": 2,
        "frequencies": {
            "A": 1,
            "B": 3,
        },
    }
