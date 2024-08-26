# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import csv
import json
import os

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


FILE_CONTENT = """\
    Text data.
    Second line of data."""


@pytest.fixture(scope="session")
def text_file(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "file.txt")
    data = bytes(FILE_CONTENT, "utf-8")
    with open(path, "wb") as f:
        f.write(data)
    return path


@pytest.fixture(scope="session")
def gz_file(tmp_path_factory: pytest.TempPathFactory) -> str:
    import gzip

    path = str(tmp_path_factory.mktemp("data") / "file.txt.gz")
    data = bytes(FILE_CONTENT, "utf-8")
    with gzip.open(path, "wb") as f:
        f.write(data)
    return path


@pytest.fixture(scope="session")
def zip_file(tmp_path_factory: pytest.TempPathFactory, text_file: str) -> str:
    import zipfile

    path = str(tmp_path_factory.mktemp("data") / "file.txt.zip")
    with zipfile.ZipFile(path, "w") as f:
        f.write(text_file, arcname=os.path.basename(text_file))
    return path


@pytest.fixture(scope="session")
def tar_file(tmp_path_factory: pytest.TempPathFactory, text_file: str) -> str:
    import tarfile

    path = str(tmp_path_factory.mktemp("data") / "file.txt.tar")
    with tarfile.TarFile(path, "w") as f:
        f.add(text_file, arcname=os.path.basename(text_file))
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


@pytest.fixture(scope="session")
def n_configs_paths(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    directory = tmp_path_factory.mktemp("data")
    readme = directory / "README.md"
    N = 15
    lines = [
        "---",
        "configs:",
    ]
    for i in range(N):
        lines += [
            f"- config_name: config{i}",
            f'  data_files: "config{i}.csv"',
        ]
    lines += [
        "---",
    ]
    with open(readme, "w", newline="") as f:
        f.writelines(f"{line}\n" for line in lines)
    files = [str(readme)]
    for i in range(N):
        config_name = f"config{i}"
        path = directory / f"{config_name}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text"])
            writer.writeheader()
            for _ in range(1000):
                writer.writerow({"text": config_name})
        files.append(str(path))
    return files
