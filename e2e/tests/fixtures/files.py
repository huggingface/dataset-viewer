# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import csv

import pytest
from pytest import TempPathFactory

DATA = [
    {
        "col_1": "There goes another one.",
        "col_2": 0,
        "col_3": 0.0,
        "col_4": "B",
    },
    {
        "col_1": "Vader turns round and round in circles as his ship spins into space.",
        "col_2": 1,
        "col_3": 1.0,
        "col_4": "B",
    },
    {
        "col_1": "We count thirty Rebel ships, Lord Vader.",
        "col_2": 2,
        "col_3": 2.0,
        "col_4": "A",
    },
    {
        "col_1": "The wingman spots the pirateship coming at him and warns the Dark Lord",
        "col_2": 3,
        "col_3": 3.0,
        "col_4": "B",
    },
]


@pytest.fixture(scope="session")
def csv_path(tmp_path_factory: TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "dataset.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["col_1", "col_2", "col_3", "col_4"])
        writer.writeheader()
        for item in DATA:
            writer.writerow(item)
    return path
