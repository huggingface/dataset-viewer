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


DATASET_SCRIPT_WITH_EXTERNAL_FILES_CONTENT = """
import datasets

_URLS = {
    "train": [
        "https://huggingface.co/datasets/lhoestq/test/resolve/main/some_text.txt",
        "https://huggingface.co/datasets/lhoestq/test/resolve/main/another_text.txt",
    ]
}


class Test(datasets.GeneratorBasedBuilder):


    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                }
            ),
            homepage="https://huggingface.co/datasets/lhoestq/test",
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
        ]

    def _generate_examples(self, filepaths):
        _id = 0
        for filepath in filepaths:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    yield _id, {"text": line.rstrip()}
                    _id += 1
"""


@pytest.fixture(scope="session")
def dataset_script_with_external_files_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "{dataset_name}.py")
    with open(path, "w", newline="") as f:
        f.write(DATASET_SCRIPT_WITH_EXTERNAL_FILES_CONTENT)
    return path


DATASET_SCRIPT_WITH_TWO_CONFIGS = """
import os

import datasets
from datasets import DatasetInfo, BuilderConfig, Features, Split, SplitGenerator, Value


class DummyDataset(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [BuilderConfig(name="first"), BuilderConfig(name="second")]

    def _info(self) -> DatasetInfo:
        return DatasetInfo(features=Features({"text": Value("string")}))

    def _split_generators(self, dl_manager):
        return [
            SplitGenerator(Split.TRAIN, gen_kwargs={"text": self.config.name}),
            SplitGenerator(Split.TEST, gen_kwargs={"text": self.config.name}),
        ]

    def _generate_examples(self, text, **kwargs):
        for i in range(1000):
            yield i, {"text": text}
"""


@pytest.fixture(scope="session")
def dataset_script_with_two_configs_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "{dataset_name}.py")
    with open(path, "w", newline="") as f:
        f.write(DATASET_SCRIPT_WITH_TWO_CONFIGS)
    return path


# N = 15
DATASET_SCRIPT_WITH_N_CONFIGS = """
import os

import datasets
from datasets import DatasetInfo, BuilderConfig, Features, Split, SplitGenerator, Value


class DummyDataset(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [BuilderConfig(name="config"+str(i)) for i in range(15)]

    def _info(self) -> DatasetInfo:
        return DatasetInfo(features=Features({"text": Value("string")}))

    def _split_generators(self, dl_manager):
        return [
            SplitGenerator(Split.TRAIN, gen_kwargs={"text": self.config.name}),
        ]

    def _generate_examples(self, text, **kwargs):
        for i in range(1000):
            yield i, {"text": text}
"""


@pytest.fixture(scope="session")
def dataset_script_with_n_configs_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "{dataset_name}.py")
    with open(path, "w", newline="") as f:
        f.write(DATASET_SCRIPT_WITH_N_CONFIGS)
    return path


DATASET_SCRIPT_WITH_MANUAL_DOWNLOAD = """
import os

import datasets
from datasets import DatasetInfo, BuilderConfig, Features, Split, SplitGenerator, Value


class DummyDatasetManualDownload(datasets.GeneratorBasedBuilder):

    @property
    def manual_download_instructions(self):
        return "To use DummyDatasetManualDownload you have to download it manually."

    def _info(self) -> DatasetInfo:
        return DatasetInfo(features=Features({"text": Value("string")}))

    def _split_generators(self, dl_manager):
        return [
            SplitGenerator(Split.TRAIN, gen_kwargs={"text": self.config.name}),
        ]

    def _generate_examples(self, text, **kwargs):
        for i in range(1000):
            yield i, {"text": text}
"""


@pytest.fixture(scope="session")
def dataset_script_with_manual_download_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "{dataset_name}.py")
    with open(path, "w", newline="") as f:
        f.write(DATASET_SCRIPT_WITH_MANUAL_DOWNLOAD)
    return path
