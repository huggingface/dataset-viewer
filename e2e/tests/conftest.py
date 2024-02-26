# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import csv
from collections.abc import Iterator
from pathlib import Path

import pytest
from pytest import TempPathFactory

from .constants import DATA, NORMAL_USER, NORMAL_USER_TOKEN
from .utils import poll, tmp_dataset


@pytest.fixture(autouse=True, scope="session")
def ensure_services_are_up() -> None:
    assert poll("/", expected_code=404).status_code == 404


@pytest.fixture(scope="session")
def csv_path(tmp_path_factory: TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "dataset.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["col_1", "col_2", "col_3", "col_4", "col_5", "col_6"])
        writer.writeheader()
        for item in DATA:
            writer.writerow(item)
    return path


@pytest.fixture(scope="session")
def normal_user_public_dataset(csv_path: str) -> Iterator[str]:
    with tmp_dataset(namespace=NORMAL_USER, token=NORMAL_USER_TOKEN, files={"data/csv_data.csv": csv_path}) as dataset:
        yield dataset


@pytest.fixture(scope="session")
def normal_user_images_public_dataset() -> Iterator[str]:
    with tmp_dataset(
        namespace=NORMAL_USER,
        token=NORMAL_USER_TOKEN,
        files={
            "1.jpg": str(Path(__file__).resolve().parent / "data" / "images" / "1.jpg"),
            "2.jpg": str(Path(__file__).resolve().parent / "data" / "images" / "2.jpg"),
            "metadata.csv": str(Path(__file__).resolve().parent / "data" / "images" / "metadata.csv"),
        },
    ) as dataset:
        yield dataset


@pytest.fixture(scope="session")
def normal_user_audios_public_dataset() -> Iterator[str]:
    with tmp_dataset(
        namespace=NORMAL_USER,
        token=NORMAL_USER_TOKEN,
        files={
            "1.wav": str(Path(__file__).resolve().parent / "data" / "audios" / "1.wav"),
            "2.wav": str(Path(__file__).resolve().parent / "data" / "audios" / "2.wav"),
            "metadata.csv": str(Path(__file__).resolve().parent / "data" / "audios" / "metadata.csv"),
        },
    ) as dataset:
        yield dataset
