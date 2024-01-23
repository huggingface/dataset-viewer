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

from ..constants import (
    ASSETS_BASE_URL,
    DEFAULT_COLUMN_NAME,
    DEFAULT_CONFIG,
    DEFAULT_REVISION,
    DEFAULT_ROW_IDX,
    DEFAULT_SAMPLING_RATE,
    DEFAULT_SPLIT,
)
from ..types import DatasetFixture


def value(content: Any, dtype: Any) -> Dataset:
    return Dataset.from_pandas(pd.DataFrame({DEFAULT_COLUMN_NAME: [content]}, dtype=dtype))


def other(content: Any, feature_type: Optional[FeatureType] = None) -> Dataset:
    if feature_type:
        features = Features({DEFAULT_COLUMN_NAME: feature_type})
        return Dataset.from_dict({DEFAULT_COLUMN_NAME: [content]}, features=features)
    else:
        return Dataset.from_dict({DEFAULT_COLUMN_NAME: [content]})


@pytest.fixture(scope="session")
def datasets_fixtures() -> Mapping[str, DatasetFixture]:
    return {
        dataset_name: DatasetFixture(
            dataset=dataset, expected_feature_type=expected_feature_type, expected_row=expected_row
        )
        for (dataset_name, dataset, expected_feature_type, expected_row) in [
            (
                "null",
                value(None, None),
                {"_type": "Value", "dtype": "null"},
                {DEFAULT_COLUMN_NAME: None},
            ),
            (
                "bool",
                value(False, pd.BooleanDtype()),
                {"_type": "Value", "dtype": "bool"},
                {DEFAULT_COLUMN_NAME: False},
            ),
            (
                "int8",
                value(-7, pd.Int8Dtype()),
                {"_type": "Value", "dtype": "int8"},
                {DEFAULT_COLUMN_NAME: -7},
            ),
            (
                "int16",
                value(-7, pd.Int16Dtype()),
                {"_type": "Value", "dtype": "int16"},
                {DEFAULT_COLUMN_NAME: -7},
            ),
            (
                "int32",
                value(-7, pd.Int32Dtype()),
                {"_type": "Value", "dtype": "int32"},
                {DEFAULT_COLUMN_NAME: -7},
            ),
            (
                "int64",
                value(-7, pd.Int64Dtype()),
                {"_type": "Value", "dtype": "int64"},
                {DEFAULT_COLUMN_NAME: -7},
            ),
            (
                "uint8",
                value(7, pd.UInt8Dtype()),
                {"_type": "Value", "dtype": "uint8"},
                {DEFAULT_COLUMN_NAME: 7},
            ),
            (
                "uint16",
                value(7, pd.UInt16Dtype()),
                {"_type": "Value", "dtype": "uint16"},
                {DEFAULT_COLUMN_NAME: 7},
            ),
            (
                "uint32",
                value(7, pd.UInt32Dtype()),
                {"_type": "Value", "dtype": "uint32"},
                {DEFAULT_COLUMN_NAME: 7},
            ),
            (
                "uint64",
                value(7, pd.UInt64Dtype()),
                {"_type": "Value", "dtype": "uint64"},
                {DEFAULT_COLUMN_NAME: 7},
            ),
            (
                "float16",
                value(-3.14, np.float16),
                {"_type": "Value", "dtype": "float16"},
                {DEFAULT_COLUMN_NAME: np.float16(-3.14)},
            ),
            (
                "float32",
                value(-3.14, np.float32),
                {"_type": "Value", "dtype": "float32"},
                {DEFAULT_COLUMN_NAME: np.float32(-3.14)},
            ),
            (
                "float64",
                value(-3.14, np.float64),
                {"_type": "Value", "dtype": "float64"},
                {DEFAULT_COLUMN_NAME: np.float64(-3.14)},
            ),
            (
                "time",
                value(datetime.time(1, 1, 1), None),
                {"_type": "Value", "dtype": "time64[us]"},
                {DEFAULT_COLUMN_NAME: datetime.time(1, 1, 1)},
            ),
            (
                "timestamp_1",
                value(pd.Timestamp(2020, 1, 1), None),
                {"_type": "Value", "dtype": "timestamp[ns]"},
                {DEFAULT_COLUMN_NAME: pd.Timestamp("2020-01-01 00:00:00")},
            ),
            (
                "timestamp_2",
                value(pd.Timestamp(1513393355.5, unit="s"), None),
                {"_type": "Value", "dtype": "timestamp[ns]"},
                {DEFAULT_COLUMN_NAME: pd.Timestamp(1513393355.5, unit="s")},
            ),
            (
                "timestamp_3",
                value(pd.Timestamp(1513393355500, unit="ms"), None),
                {"_type": "Value", "dtype": "timestamp[ns]"},
                {DEFAULT_COLUMN_NAME: pd.Timestamp(1513393355500, unit="ms")},
            ),
            (
                "timestamp_tz",
                value(pd.Timestamp(year=2020, month=1, day=1, tz="US/Pacific"), None),
                {"_type": "Value", "dtype": "timestamp[ns, tz=US/Pacific]"},
                {DEFAULT_COLUMN_NAME: pd.Timestamp(year=2020, month=1, day=1, tz="US/Pacific")},
            ),
            (
                "string",
                value("a string", pd.StringDtype(storage="python")),
                {"_type": "Value", "dtype": "string"},
                {DEFAULT_COLUMN_NAME: "a string"},
            ),
            # other types of features
            (
                "class_label",
                other("positive", ClassLabel(names=["negative", "positive"])),
                {"_type": "ClassLabel", "names": ["negative", "positive"]},
                {DEFAULT_COLUMN_NAME: 1},
            ),
            (
                "dict",
                other({"a": 0}, None),
                {"a": {"_type": "Value", "dtype": "int64"}},
                {DEFAULT_COLUMN_NAME: {"a": 0}},
            ),
            (
                "list",
                other([{"a": 0}], None),
                [{"a": {"_type": "Value", "dtype": "int64"}}],
                {DEFAULT_COLUMN_NAME: [{"a": 0}]},
            ),
            (
                "sequence_simple",
                other([0], None),
                {"_type": "Sequence", "feature": {"_type": "Value", "dtype": "int64"}},
                {DEFAULT_COLUMN_NAME: [0]},
            ),
            (
                "sequence",
                other([{"a": 0}], Sequence(feature={"a": Value(dtype="int64")})),
                {"_type": "Sequence", "feature": {"a": {"_type": "Value", "dtype": "int64"}}},
                {DEFAULT_COLUMN_NAME: {"a": [0]}},
            ),
            (
                "array2d",
                other(np.zeros((2, 2), dtype="float32"), Array2D(shape=(2, 2), dtype="float32")),
                {"_type": "Array2D", "shape": (2, 2), "dtype": "float32"},
                {DEFAULT_COLUMN_NAME: [[0.0, 0.0], [0.0, 0.0]]},
            ),
            (
                "array3d",
                other(np.zeros((2, 2, 2), dtype="float32"), Array3D(shape=(2, 2, 2), dtype="float32")),
                {"_type": "Array3D", "shape": (2, 2, 2), "dtype": "float32"},
                {DEFAULT_COLUMN_NAME: [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]},
            ),
            (
                "array4d",
                other(np.zeros((2, 2, 2, 2), dtype="float32"), Array4D(shape=(2, 2, 2, 2), dtype="float32")),
                {"_type": "Array4D", "shape": (2, 2, 2, 2), "dtype": "float32"},
                {
                    DEFAULT_COLUMN_NAME: [
                        [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                        [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                    ]
                },
            ),
            (
                "array5d",
                other(np.zeros((2, 2, 2, 2, 2), dtype="float32"), Array5D(shape=(2, 2, 2, 2, 2), dtype="float32")),
                {"_type": "Array5D", "shape": (2, 2, 2, 2, 2), "dtype": "float32"},
                {
                    DEFAULT_COLUMN_NAME: [
                        [
                            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                        ],
                        [
                            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                        ],
                    ]
                },
            ),
            (
                "translation",
                other({"en": "the cat", "fr": "le chat"}, Translation(languages=["en", "fr"])),
                {"_type": "Translation", "languages": ["en", "fr"]},
                {DEFAULT_COLUMN_NAME: {"en": "the cat", "fr": "le chat"}},
            ),
            (
                "translation_variable_languages",
                other(
                    {"en": "the cat", "fr": ["le chat", "la chatte"]},
                    TranslationVariableLanguages(languages=["en", "fr"]),
                ),
                {
                    "_type": "TranslationVariableLanguages",
                    "languages": ["en", "fr"],
                    "num_languages": 2,
                },
                {
                    DEFAULT_COLUMN_NAME: {
                        "language": ["en", "fr", "fr"],
                        "translation": ["the cat", "la chatte", "le chat"],
                    }
                },
            ),
            (
                "audio",
                other(
                    {"array": [0.1, 0.2, 0.3], "sampling_rate": DEFAULT_SAMPLING_RATE},
                    Audio(sampling_rate=DEFAULT_SAMPLING_RATE),
                ),
                {
                    "_type": "Audio",
                    "sampling_rate": DEFAULT_SAMPLING_RATE,  # <- why do we have a sampling rate here? (see below for contrast)
                },
                {
                    DEFAULT_COLUMN_NAME: [
                        {
                            "src": f"{ASSETS_BASE_URL}/audio/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/audio.wav",
                            "type": "audio/wav",
                        }
                    ]
                },
            ),
            (
                "audio_ogg",
                other(
                    str(Path(__file__).resolve().parent / "data" / "test_audio_vorbis.ogg"),
                    Audio(sampling_rate=DEFAULT_SAMPLING_RATE),
                ),
                {
                    "_type": "Audio",
                    "sampling_rate": DEFAULT_SAMPLING_RATE,
                },
                {
                    DEFAULT_COLUMN_NAME: [
                        {
                            "src": f"{ASSETS_BASE_URL}/audio_ogg/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/audio.wav",
                            "type": "audio/wav",
                        }
                    ]
                },
            ),
            (
                "image",
                other(
                    str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
                    Image(),
                ),
                {"_type": "Image"},
                {
                    DEFAULT_COLUMN_NAME: {
                        "src": f"{ASSETS_BASE_URL}/image/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/image.jpg",
                        "height": 480,
                        "width": 640,
                    }
                },
            ),
            (
                "images_list",
                other(
                    [
                        str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
                        str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
                    ],
                    [Image()],
                ),
                [
                    {
                        "_type": "Image",
                    }
                ],
                {
                    DEFAULT_COLUMN_NAME: [
                        {
                            "src": f"{ASSETS_BASE_URL}/images_list/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/image-1d100e9.jpg",
                            # ^ suffix -1d100e9 comes from append_hash_suffix() to get a unique filename
                            "height": 480,
                            "width": 640,
                        },
                        {
                            "src": f"{ASSETS_BASE_URL}/images_list/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/image-1d300ea.jpg",
                            "height": 480,
                            "width": 640,
                        },
                    ]
                },
            ),
            (
                "audios_list",
                other(
                    [
                        {"array": [0.1, 0.2, 0.3], "sampling_rate": DEFAULT_SAMPLING_RATE},
                        {"array": [0.1, 0.2, 0.3], "sampling_rate": DEFAULT_SAMPLING_RATE},
                    ],
                    [Audio()],
                ),
                [
                    {
                        "_type": "Audio",  # <- why don't we have a sampling rate here?
                    }
                ],
                {
                    DEFAULT_COLUMN_NAME: [
                        [
                            {
                                "src": f"{ASSETS_BASE_URL}/audios_list/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/audio-1d100e9.wav",
                                "type": "audio/wav",
                            }
                        ],
                        [
                            {
                                "src": f"{ASSETS_BASE_URL}/audios_list/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/audio-1d300ea.wav",
                                "type": "audio/wav",
                            },
                        ],
                    ]
                },
            ),
            (
                "images_sequence",
                other(
                    [
                        str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
                        str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
                    ],
                    Sequence(feature=Image()),
                ),
                {
                    "_type": "Sequence",
                    "feature": {
                        "_type": "Image",
                    },
                },
                {
                    DEFAULT_COLUMN_NAME: [
                        {
                            "src": f"{ASSETS_BASE_URL}/images_sequence/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/image-1d100e9.jpg",
                            "height": 480,
                            "width": 640,
                        },
                        {
                            "src": f"{ASSETS_BASE_URL}/images_sequence/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/image-1d300ea.jpg",
                            "height": 480,
                            "width": 640,
                        },
                    ]
                },
            ),
            (
                "audios_sequence",
                other(
                    [
                        {"array": [0.1, 0.2, 0.3], "sampling_rate": DEFAULT_SAMPLING_RATE},
                        {"array": [0.1, 0.2, 0.3], "sampling_rate": DEFAULT_SAMPLING_RATE},
                    ],
                    Sequence(feature=Audio()),
                ),
                {
                    "_type": "Sequence",
                    "feature": {
                        "_type": "Audio",  # <- why don't we have a sampling rate here?
                    },
                },
                {
                    DEFAULT_COLUMN_NAME: [
                        [
                            {
                                "src": f"{ASSETS_BASE_URL}/audios_sequence/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/audio-1d100e9.wav",
                                "type": "audio/wav",
                            }
                        ],
                        [
                            {
                                "src": f"{ASSETS_BASE_URL}/audios_sequence/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/audio-1d300ea.wav",
                                "type": "audio/wav",
                            }
                        ],
                    ]
                },
            ),
            (
                "dict_of_audios_and_images",
                other(
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
                {
                    "a": {"_type": "Value", "dtype": "int64"},
                    "b": [{"_type": "Image"}],
                    "c": {"ca": [{"_type": "Audio"}]},  # <- why don't we have a sampling rate here?
                },
                {
                    DEFAULT_COLUMN_NAME: {
                        "a": 0,
                        "b": [
                            {
                                "src": f"{ASSETS_BASE_URL}/dict_of_audios_and_images/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/image-89101db.jpg",
                                "height": 480,
                                "width": 640,
                            },
                            {
                                "src": f"{ASSETS_BASE_URL}/dict_of_audios_and_images/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/image-89301dc.jpg",
                                "height": 480,
                                "width": 640,
                            },
                        ],
                        "c": {
                            "ca": [
                                [
                                    {
                                        "src": f"{ASSETS_BASE_URL}/dict_of_audios_and_images/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/audio-18360330.wav",
                                        "type": "audio/wav",
                                    }
                                ],
                                [
                                    {
                                        "src": f"{ASSETS_BASE_URL}/dict_of_audios_and_images/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/audio-18380331.wav",
                                        "type": "audio/wav",
                                    }
                                ],
                            ]
                        },
                    }
                },
            ),
            (
                "sequence_of_dicts",
                other([{"a": {"b": 0}}, {"a": {"b": 1}}], Sequence(feature={"a": {"b": Value(dtype="int64")}})),
                {
                    "_type": "Sequence",
                    "feature": {
                        "a": {"b": {"_type": "Value", "dtype": "int64"}},
                    },
                },
                {DEFAULT_COLUMN_NAME: {"a": [{"b": 0}, {"b": 1}]}},
            ),
            (
                "none_value",
                other({"a": None}, {"a": Value(dtype="int64")}),
                {"a": {"_type": "Value", "dtype": "int64"}},
                {DEFAULT_COLUMN_NAME: {"a": None}},
            ),
            (
                "big",
                Dataset.from_pandas(
                    pd.DataFrame(
                        {DEFAULT_COLUMN_NAME: ["a" * 1_234 for _ in range(4_567)]},
                        dtype=pd.StringDtype(storage="python"),
                    )
                ),
                {"_type": "Value", "dtype": "string"},
                {DEFAULT_COLUMN_NAME: "a" * 1_234},
            ),
        ]
    }
