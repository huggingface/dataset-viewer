# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from .utils import get_default_config_split, poll_until_ready_and_assert


def test_statistics_endpoint(normal_user_public_json_dataset: str) -> None:
    # TODO: add dataset with various splits, or various configs
    dataset = normal_user_public_json_dataset
    config, split = get_default_config_split()
    statistics_response = poll_until_ready_and_assert(
        relative_url=f"/statistics?dataset={dataset}&config={config}&split={split}",
        check_x_revision=True,
        dataset=dataset,
    )

    content = statistics_response.json()
    assert len(content) == 3
    assert sorted(content) == ["num_examples", "partial", "statistics"], statistics_response
    statistics = content["statistics"]
    num_examples = content["num_examples"]
    partial = content["partial"]

    assert isinstance(statistics, list), statistics
    assert len(statistics) == 6
    assert num_examples == 4
    assert partial is False

    string_label_column = statistics[0]
    assert "column_name" in string_label_column
    assert "column_statistics" in string_label_column
    assert "column_type" in string_label_column
    assert string_label_column["column_name"] == "col_1"
    assert string_label_column["column_type"] == "string_label"
    assert isinstance(string_label_column["column_statistics"], dict)
    assert string_label_column["column_statistics"] == {
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

    int_column = statistics[1]
    assert "column_name" in int_column
    assert "column_statistics" in int_column
    assert "column_type" in int_column
    assert int_column["column_name"] == "col_2"
    assert int_column["column_type"] == "int"
    assert isinstance(int_column["column_statistics"], dict)
    assert int_column["column_statistics"] == {
        "histogram": {"bin_edges": [0, 1, 2, 3, 3], "hist": [1, 1, 1, 1]},
        "max": 3,
        "mean": 1.5,
        "median": 1.5,
        "min": 0,
        "nan_count": 0,
        "nan_proportion": 0.0,
        "std": 1.29099,
    }

    float_column = statistics[2]
    assert "column_name" in float_column
    assert "column_statistics" in float_column
    assert "column_type" in float_column
    assert float_column["column_name"] == "col_3"
    assert float_column["column_type"] == "float"
    assert isinstance(float_column["column_statistics"], dict)
    assert float_column["column_statistics"] == {
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

    string_label_column2 = statistics[3]
    assert "column_name" in string_label_column2
    assert "column_statistics" in string_label_column2
    assert "column_type" in string_label_column2
    assert string_label_column2["column_name"] == "col_4"
    assert string_label_column2["column_type"] == "string_label"
    assert isinstance(string_label_column2["column_statistics"], dict)
    assert string_label_column2["column_statistics"] == {
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

    bool_column = statistics[4]
    assert "column_name" in bool_column
    assert "column_statistics" in bool_column
    assert "column_type" in bool_column
    assert bool_column["column_name"] == "col_5"
    assert bool_column["column_type"] == "bool"
    assert isinstance(bool_column["column_statistics"], dict)
    assert bool_column["column_statistics"] == {
        "nan_count": 1,
        "nan_proportion": 0.25,
        "frequencies": {
            "True": 2,
            "False": 1,
        },
    }

    list_column = statistics[5]
    assert "column_name" in list_column
    assert "column_statistics" in list_column
    assert "column_type" in list_column
    assert list_column["column_name"] == "col_6"
    assert list_column["column_type"] == "list"
    assert isinstance(list_column["column_statistics"], dict)
    assert list_column["column_statistics"] == {
        "nan_count": 1,
        "nan_proportion": 0.25,
        "min": 3,
        "max": 5,
        "mean": 4.0,
        "median": 4,
        "std": 1.0,
        "histogram": {
            "hist": [1, 1, 1],
            "bin_edges": [3, 4, 5, 5],
        },
    }
