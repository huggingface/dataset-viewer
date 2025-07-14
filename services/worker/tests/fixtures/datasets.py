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
    List,
    Translation,
    TranslationVariableLanguages,
    Value,
)
from datasets.features.features import FeatureType

from .statistics_dataset import (
    audio_dataset,
    datetime_dataset,
    image_dataset,
    null_column,
    statistics_dataset,
    statistics_not_supported_dataset,
    statistics_string_text_dataset,
)

SEARCH_TEXT_CONTENT = [
    ("Grand Moff Tarkin and Lord Vader are interrupted in their discussion by the buzz of the" " comlink"),
    "There goes another one.",
    "Vader turns round and round in circles as his ship spins into space.",
    "We count thirty Rebel ships, Lord Vader.",
    "The wingman spots the pirateship coming at him and warns the Dark Lord",
]


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
        "sequence": other([{"a": 0}], List({"a": Value(dtype="int64")})),
        "array2d": other(np.zeros((2, 2), dtype="float32"), Array2D(shape=(2, 2), dtype="float32")),
        "array3d": other(np.zeros((2, 2, 2), dtype="float32"), Array3D(shape=(2, 2, 2), dtype="float32")),
        "array4d": other(np.zeros((2, 2, 2, 2), dtype="float32"), Array4D(shape=(2, 2, 2, 2), dtype="float32")),
        "array5d": other(np.zeros((2, 2, 2, 2, 2), dtype="float32"), Array5D(shape=(2, 2, 2, 2, 2), dtype="float32")),
        "audio": other(
            str(
                Path(__file__).resolve().parent.parent.parent.parent.parent
                / "libs"
                / "libcommon"
                / "tests"
                / "viewer_utils"
                / "data"
                / "test_audio_16000.mp3"
            ),
            Audio(sampling_rate=sampling_rate),
        ),
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
            List(Image()),
        ),
        "audios_list": other(
            [
                str(Path(__file__).resolve().parent.parent / "viewer_utils" / "data" / "test_audio_16000.mp3"),
                str(Path(__file__).resolve().parent.parent / "viewer_utils" / "data" / "test_audio_16000.mp3"),
            ],
            List(Audio()),
        ),
        "images_sequence": other(
            [
                str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
                str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
            ],
            List(feature=Image()),
        ),
        "audios_sequence": other(
            [
                str(Path(__file__).resolve().parent.parent / "viewer_utils" / "data" / "test_audio_16000.mp3"),
                str(Path(__file__).resolve().parent.parent / "viewer_utils" / "data" / "test_audio_16000.mp3"),
            ],
            List(feature=Audio()),
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
                        str(Path(__file__).resolve().parent.parent / "viewer_utils" / "data" / "test_audio_16000.mp3"),
                        str(Path(__file__).resolve().parent.parent / "viewer_utils" / "data" / "test_audio_16000.mp3"),
                    ]
                },
            },
            {"a": Value(dtype="int64"), "b": List(Image()), "c": {"ca": List(Audio())}},
        ),
        "sequence_of_dicts": other(
            [{"a": {"b": 0}}, {"a": {"b": 1}}], List(feature={"a": {"b": Value(dtype="int64")}})
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
                        None,
                    ]
                },
                dtype=pd.StringDtype(storage="python"),
            )
        ),
        "presidio_scan": Dataset.from_pandas(
            pd.DataFrame(
                {
                    "col": [
                        "My name is Giovanni Giorgio",
                        "but everyone calls me Giorgio",
                        "My IP address is 192.168.0.1",
                        "My SSN is 345-67-8901",
                        "My email is giovanni.giorgio@daftpunk.com",
                        None,
                    ]
                },
                dtype=pd.StringDtype(storage="python"),
            )
        ),
        "duckdb_index": Dataset.from_dict(
            {
                "text": SEARCH_TEXT_CONTENT,
                "text_all_null": null_column(5),
                "column with spaces": [
                    "a",
                    "b",
                    "c",
                    "d",
                    "e",
                ],
                "list": [
                    [1],
                    [1, 2],
                    None,
                    [1, 2, 3, 4],
                    [1, 2, 3, 4, 5],
                ],
                "list_all_null": null_column(5),
                "list_struct": [
                    [],
                    [{"author": "cat", "likes": 5}],
                    [{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}],
                    [{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}],
                    None,
                ],
                "audio": list(audio_dataset["audio"]) + [None],
                "audio_all_null": null_column(5),
                "image": list(image_dataset["image"]) + [None],
                "image_all_null": null_column(5),
            },
            features=Features(
                {
                    "text": Value(dtype="string"),
                    "text_all_null": Value(dtype="string"),
                    "column with spaces": Value(dtype="string"),
                    "list": List(Value(dtype="int32")),
                    "list_all_null": List(Value(dtype="int32")),
                    "list_struct": List({"author": Value("string"), "likes": Value("int32")}),
                    "audio": Audio(sampling_rate=1600, decode=False),
                    "audio_all_null": Audio(sampling_rate=1600, decode=False),
                    "image": Image(decode=False),
                    "image_all_null": Image(decode=False),
                }
            ),
        ),
        "descriptive_statistics": statistics_dataset,
        "descriptive_statistics_string_text": statistics_string_text_dataset,
        "descriptive_statistics_not_supported": statistics_not_supported_dataset,
        "audio_statistics": audio_dataset,
        "image_statistics": image_dataset,
        "datetime_statistics": datetime_dataset,
    }
