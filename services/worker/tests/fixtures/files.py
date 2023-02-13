# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import csv
import json

import pandas as pd
import pytest

DATA = [
    {"col_1": "0", "col_2": 0, "col_3": 0.0},
    {"col_1": "1", "col_2": 1, "col_3": 1.0},
    {"col_1": "2", "col_2": 2, "col_3": 2.0},
    {"col_1": "3", "col_2": 3, "col_3": 3.0},
]


@pytest.fixture(scope="session")
def csv_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "dataset.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["col_1", "col_2", "col_3"])
        writer.writeheader()
        for item in DATA:
            writer.writerow(item)
    return path


@pytest.fixture(scope="session")
def data_df(csv_path: str) -> pd.DataFrame:
    # from the CSV file, not the DATA variable, because the CSV file does not respect the first column type
    # we have to follow the same behavior
    return pd.read_csv(csv_path)


JSONL = [
    {"col_1": "0", "col_2": 0, "col_3": 0.0},
    {"col_1": None, "col_2": 1, "col_3": 1.0},
    {"col_2": 2, "col_3": 2.0},
    {"col_1": "3", "col_2": 3, "col_3": 3.0},
]


@pytest.fixture(scope="session")
def jsonl_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "dataset.jsonl")
    with open(path, "w", newline="") as f:
        f.writelines(json.dumps(o) for o in JSONL)
    return path


@pytest.fixture(scope="session")
def extra_fields_readme(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "README.md")
    lines = [
        "---",
        'extra_gated_prompt: "You agree not to attempt to determine the identity of individuals in this dataset"',
        "extra_gated_fields:",
        "  Company: text",
        "  Country: text",
        "  I agree to use this model for non-commercial use ONLY: checkbox",
        "---",
    ]
    with open(path, "w", newline="") as f:
        f.writelines(f"{line}\n" for line in lines)
    return path
