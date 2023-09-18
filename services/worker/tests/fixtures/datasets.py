# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import datetime
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

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


def other(content: Any, feature_type: Optional[FeatureType] = None) -> Dataset:
    if feature_type:
        features = Features({"col": feature_type})
        return Dataset.from_dict({"col": [content]}, features=features)
    else:
        return Dataset.from_dict({"col": [content]})


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
        "spawning_opt_in_out": Dataset.from_pandas(
            pd.DataFrame(
                {
                    "col": [
                        "http://testurl.test/test_image-optOut.jpg",
                        "http://testurl.test/test_image2.jpg",
                        "other",
                        "http://testurl.test/test_image3-optIn.png",
                    ]
                },
                dtype=pd.StringDtype(storage="python"),
            )
        ),
        "duckdb_index": Dataset.from_pandas(
            pd.DataFrame(
                {
                    "text": [
                        (
                            "Grand Moff Tarkin and Lord Vader are interrupted in their discussion by the buzz of the"
                            " comlink"
                        ),
                        "There goes another one.",
                        "Vader turns round and round in circles as his ship spins into space.",
                        "We count thirty Rebel ships, Lord Vader.",
                        "The wingman spots the pirateship coming at him and warns the Dark Lord",
                    ],
                    "column with spaces": [
                        "a",
                        "b",
                        "c",
                        "d",
                        "e",
                    ],
                },
                dtype=pd.StringDtype(storage="python"),
            )
        ),
        "descriptive_statistics": Dataset.from_dict(
            {
                "int_column": [0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 5, 5, 5, 6, 7, 8, 8, 8],
                "int_nan_column": [0, None, 1, None, 2, None, 2, None, 4, None, 5, None, 5, 5, 5, 6, 7, 8, 8, 8],
                "float_column": [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    1.1,
                    2.2,
                    2.3,
                    2.6,
                    4.7,
                    5.1,
                    6.2,
                    6.7,
                    6.8,
                    7.0,
                    8.3,
                    8.4,
                    9.2,
                    9.7,
                    9.9,
                ],
                "float_nan_column": [
                    None,
                    0.2,
                    0.3,
                    None,
                    0.5,
                    None,
                    2.2,
                    None,
                    2.6,
                    4.7,
                    5.1,
                    None,
                    None,
                    None,
                    None,
                    8.3,
                    8.4,
                    9.2,
                    9.7,
                    9.9,
                ],
                "class_label_column": [
                    "cat",
                    "dog",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "dog",
                    "cat",
                    "dog",
                    "cat",
                ],
                "class_label_nan_column": [
                    "cat",
                    None,
                    "cat",
                    "cat",
                    "cat",
                    None,
                    "cat",
                    "cat",
                    "cat",
                    None,
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "dog",
                    "cat",
                    None,
                    "cat",
                ],
                "float_negative_column": [
                    -7.221,
                    -5.333,
                    -15.154,
                    -15.392,
                    -15.054,
                    -10.176,
                    -10.072,
                    -10.59,
                    -6.0,
                    -14.951,
                    -14.054,
                    -9.706,
                    -7.053,
                    -10.072,
                    -15.054,
                    -12.777,
                    -12.233,
                    -13.54,
                    -14.376,
                    -15.754,
                ],
                "float_cross_zero_column": [
                    -7.221,
                    -5.333,
                    -15.154,
                    -15.392,
                    -15.054,
                    -10.176,
                    -10.072,
                    -10.59,
                    6.0,
                    14.951,
                    14.054,
                    -9.706,
                    7.053,
                    0.0,
                    -15.054,
                    -12.777,
                    12.233,
                    13.54,
                    -14.376,
                    15.754,
                ],
                "float_large_values_column": [
                    1101.34567,
                    1178.923,
                    197.2134,
                    1150.8483,
                    169.907655,
                    156.4580,
                    134.4368456,
                    189.456,
                    145.0912,
                    148.354465,
                    190.8943,
                    1134.34,
                    155.22366,
                    153.0,
                    163.0,
                    143.5468,
                    177.231,
                    132.568,
                    191.99,
                    1114.0,
                ],
                "int_negative_column": [
                    -10,
                    -9,
                    -8,
                    -1,
                    -5,
                    -1,
                    -2,
                    -3,
                    -5,
                    -4,
                    -7,
                    -8,
                    -11,
                    -15,
                    -20 - 11,
                    -1,
                    -14,
                    -11,
                    -0,
                    -10,
                ],
                "int_cross_zero_column": [
                    -10,
                    -9,
                    -8,
                    0,
                    0,
                    1,
                    2,
                    -3,
                    -5,
                    4,
                    7,
                    8,
                    11,
                    15,
                    20 - 11,
                    -1,
                    14,
                    11,
                    0,
                    -10,
                ],
                "int_large_values_column": [
                    1101,
                    1178,
                    197,
                    1150,
                    169,
                    156,
                    134,
                    189,
                    145,
                    148,
                    190,
                    1134,
                    155,
                    153,
                    163,
                    143,
                    177,
                    132,
                    191,
                    1114,
                ],
            },
            features=Features(
                {
                    "int_column": Value("int32"),
                    "int_nan_column": Value("int32"),
                    "int_negative_column": Value("int32"),
                    "int_cross_zero_column": Value("int32"),
                    "int_large_values_column": Value("int32"),
                    "float_column": Value("float32"),
                    "float_nan_column": Value("float32"),
                    "float_negative_column": Value("float64"),
                    "float_cross_zero_column": Value("float32"),
                    "float_large_values_column": Value("float32"),
                    "class_label_column": ClassLabel(names=["cat", "dog"]),
                    "class_label_nan_column": ClassLabel(names=["cat", "dog"]),
                }
            ),
        ),
    }
