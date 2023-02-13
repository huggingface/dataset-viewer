# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest
from datasets import (
    Array2D,
    Array3D,
    Array4D,
    Array5D,
    Audio,
    ClassLabel,
    Dataset,
    Features,
    Image,
    Sequence,
    Translation,
    TranslationVariableLanguages,
    Value,
)
from datasets.features.features import FeatureType


def value(content: Any, dtype: Any) -> Dataset:
    return Dataset.from_pandas(pd.DataFrame({"col": [content]}, dtype=dtype))


def other(content: Any, feature_type: FeatureType = None) -> Dataset:
    return (
        Dataset.from_dict({"col": [content]})
        if feature_type is None
        else Dataset.from_dict({"col": [content]}, features=Features({"col": feature_type}))
    )


@pytest.fixture(scope="session")
def datasets() -> Mapping[str, Dataset]:
    sampling_rate = 16_000
    return {
        # Value feature
        "null": value(None, None),
        "bool": value(False, pd.BooleanDtype()),
        "int8": value(-7, pd.Int8Dtype()),
        "int16": value(-7, pd.Int16Dtype()),
        "int32": value(-7, pd.Int32Dtype()),
        "int64": value(-7, pd.Int64Dtype()),
        "uint8": value(7, pd.UInt8Dtype()),
        "uint16": value(7, pd.UInt16Dtype()),
        "uint32": value(7, pd.UInt32Dtype()),
        "uint64": value(7, pd.UInt64Dtype()),
        "float16": value(-3.14, np.float16),
        "float32": value(-3.14, np.float32),
        "float64": value(-3.14, np.float64),
        "time": value(datetime.time(1, 1, 1), None),
        "timestamp_1": value(pd.Timestamp(2020, 1, 1), None),
        "timestamp_2": value(pd.Timestamp(1513393355.5, unit="s"), None),
        "timestamp_3": value(pd.Timestamp(1513393355500, unit="ms"), None),
        "timestamp_tz": value(pd.Timestamp(year=2020, month=1, day=1, tz="US/Pacific"), None),
        "string": value("a string", pd.StringDtype(storage="python")),
        # other types of features
        "class_label": other("positive", ClassLabel(names=["negative", "positive"])),
        "dict": other({"a": 0}, None),
        "list": other([{"a": 0}], None),
        "sequence_simple": other([0], None),
        "sequence": other([{"a": 0}], Sequence(feature={"a": Value(dtype="int64")})),
        "array2d": other(np.zeros((2, 2), dtype="float32"), Array2D(shape=(2, 2), dtype="float32")),
        "array3d": other(np.zeros((2, 2, 2), dtype="float32"), Array3D(shape=(2, 2, 2), dtype="float32")),
        "array4d": other(np.zeros((2, 2, 2, 2), dtype="float32"), Array4D(shape=(2, 2, 2, 2), dtype="float32")),
        "array5d": other(np.zeros((2, 2, 2, 2, 2), dtype="float32"), Array5D(shape=(2, 2, 2, 2, 2), dtype="float32")),
        "audio": other({"array": [0.1, 0.2, 0.3], "sampling_rate": sampling_rate}, Audio(sampling_rate=sampling_rate)),
        "image": other(str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"), Image()),
        "translation": other({"en": "the cat", "fr": "le chat"}, Translation(languages=["en", "fr"])),
        "translation_variable_languages": other(
            {"en": "the cat", "fr": ["le chat", "la chatte"]},
            TranslationVariableLanguages(languages=["en", "fr"]),
        ),
        "images_list": other(
            [
                str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
                str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
            ],
            [Image()],
        ),
        "audios_list": other(
            [
                {"array": [0.1, 0.2, 0.3], "sampling_rate": 16_000},
                {"array": [0.1, 0.2, 0.3], "sampling_rate": 16_000},
            ],
            [Audio()],
        ),
        "images_sequence": other(
            [
                str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
                str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
            ],
            Sequence(feature=Image()),
        ),
        "audios_sequence": other(
            [
                {"array": [0.1, 0.2, 0.3], "sampling_rate": 16_000},
                {"array": [0.1, 0.2, 0.3], "sampling_rate": 16_000},
            ],
            Sequence(feature=Audio()),
        ),
        "dict_of_audios_and_images": other(
            {
                "a": 0,
                "b": [
                    str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
                    str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
                ],
                "c": {
                    "ca": [
                        {"array": [0.1, 0.2, 0.3], "sampling_rate": 16_000},
                        {"array": [0.1, 0.2, 0.3], "sampling_rate": 16_000},
                    ]
                },
            },
            {"a": Value(dtype="int64"), "b": [Image()], "c": {"ca": [Audio()]}},
        ),
        "sequence_of_dicts": other(
            [{"a": {"b": 0}}, {"a": {"b": 1}}], Sequence(feature={"a": {"b": Value(dtype="int64")}})
        ),
        "none_value": other({"a": None}, {"a": Value(dtype="int64")}),
        "big": Dataset.from_pandas(
            pd.DataFrame({"col": ["a" * 1_234 for _ in range(4_567)]}, dtype=pd.StringDtype(storage="python"))
        ),
    }
