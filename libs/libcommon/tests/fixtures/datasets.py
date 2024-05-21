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

from libcommon.url_signer import AssetUrlPath

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
        "null": DatasetFixture(
            value(None, None),
            {"_type": "Value", "dtype": "null"},
            None,
            [],
            0,
        ),
        "bool": DatasetFixture(
            value(False, pd.BooleanDtype()),
            {"_type": "Value", "dtype": "bool"},
            False,
            [],
            0,
        ),
        "int8": DatasetFixture(
            value(-7, pd.Int8Dtype()),
            {"_type": "Value", "dtype": "int8"},
            -7,
            [],
            0,
        ),
        "int16": DatasetFixture(
            value(-7, pd.Int16Dtype()),
            {"_type": "Value", "dtype": "int16"},
            -7,
            [],
            0,
        ),
        "int32": DatasetFixture(
            value(-7, pd.Int32Dtype()),
            {"_type": "Value", "dtype": "int32"},
            -7,
            [],
            0,
        ),
        "int64": DatasetFixture(
            value(-7, pd.Int64Dtype()),
            {"_type": "Value", "dtype": "int64"},
            -7,
            [],
            0,
        ),
        "uint8": DatasetFixture(
            value(7, pd.UInt8Dtype()),
            {"_type": "Value", "dtype": "uint8"},
            7,
            [],
            0,
        ),
        "uint16": DatasetFixture(
            value(7, pd.UInt16Dtype()),
            {"_type": "Value", "dtype": "uint16"},
            7,
            [],
            0,
        ),
        "uint32": DatasetFixture(
            value(7, pd.UInt32Dtype()),
            {"_type": "Value", "dtype": "uint32"},
            7,
            [],
            0,
        ),
        "uint64": DatasetFixture(
            value(7, pd.UInt64Dtype()),
            {"_type": "Value", "dtype": "uint64"},
            7,
            [],
            0,
        ),
        "float16": DatasetFixture(
            value(-3.14, np.float16),
            {"_type": "Value", "dtype": "float16"},
            np.float16(-3.14),
            [],
            0,
        ),
        "float32": DatasetFixture(
            value(-3.14, np.float32),
            {"_type": "Value", "dtype": "float32"},
            np.float32(-3.14),
            [],
            0,
        ),
        "float64": DatasetFixture(
            value(-3.14, np.float64),
            {"_type": "Value", "dtype": "float64"},
            np.float64(-3.14),
            [],
            0,
        ),
        "time": DatasetFixture(
            value(datetime.time(1, 1, 1), None),
            {"_type": "Value", "dtype": "time64[us]"},
            datetime.time(1, 1, 1),
            [],
            0,
        ),
        "timestamp_1": DatasetFixture(
            value(pd.Timestamp(2020, 1, 1), None),
            {"_type": "Value", "dtype": "timestamp[ns]"},
            pd.Timestamp("2020-01-01 00:00:00"),
            [],
            0,
        ),
        "timestamp_2": DatasetFixture(
            value(pd.Timestamp(1513393355.5, unit="s"), None),
            {"_type": "Value", "dtype": "timestamp[ns]"},
            pd.Timestamp(1513393355.5, unit="s"),
            [],
            0,
        ),
        "timestamp_3": DatasetFixture(
            value(pd.Timestamp(1513393355500, unit="ms"), None),
            {"_type": "Value", "dtype": "timestamp[ns]"},
            pd.Timestamp(1513393355500, unit="ms"),
            [],
            0,
        ),
        "timestamp_tz": DatasetFixture(
            value(pd.Timestamp(year=2020, month=1, day=1, tz="US/Pacific"), None),
            {"_type": "Value", "dtype": "timestamp[ns, tz=US/Pacific]"},
            pd.Timestamp(year=2020, month=1, day=1, tz="US/Pacific"),
            [],
            0,
        ),
        "string": DatasetFixture(
            value("a string", pd.StringDtype(storage="python")),
            {"_type": "Value", "dtype": "string"},
            "a string",
            [],
            0,
        ),
        "urls": DatasetFixture(
            value("https://foo.bar", pd.StringDtype(storage="python")),
            {"_type": "Value", "dtype": "string"},
            "https://foo.bar",
            [],
            0,
        ),
        # other types of features
        "class_label": DatasetFixture(
            other("positive", ClassLabel(names=["negative", "positive"])),
            {"_type": "ClassLabel", "names": ["negative", "positive"]},
            1,
            [],
            0,
        ),
        "dict": DatasetFixture(
            other({"a": 0}, None),
            {"a": {"_type": "Value", "dtype": "int64"}},
            {"a": 0},
            [],
            0,
        ),
        "list": DatasetFixture(
            other([{"a": 0}], None),
            [{"a": {"_type": "Value", "dtype": "int64"}}],
            [{"a": 0}],
            [],
            0,
        ),
        "sequence_implicit": DatasetFixture(
            other([0], None),
            {"_type": "Sequence", "feature": {"_type": "Value", "dtype": "int64"}},
            [0],
            [],
            0,
        ),
        "sequence_list": DatasetFixture(
            other([0], Sequence(feature=Value(dtype="int64"))),
            {"_type": "Sequence", "feature": {"_type": "Value", "dtype": "int64"}},
            [0],
            [],
            0,
        ),
        "sequence_dict": DatasetFixture(
            other([{"a": 0}], Sequence(feature={"a": Value(dtype="int64")})),
            {"_type": "Sequence", "feature": {"a": {"_type": "Value", "dtype": "int64"}}},
            {"a": [0]},
            # ^ converted to a dict of lists, see https://huggingface.co/docs/datasets/v2.16.1/en/package_reference/main_classes#datasets.Features
            [],
            0,
        ),
        "array2d": DatasetFixture(
            other(np.zeros((2, 2), dtype="float32"), Array2D(shape=(2, 2), dtype="float32")),
            {"_type": "Array2D", "shape": (2, 2), "dtype": "float32"},
            [[0.0, 0.0], [0.0, 0.0]],
            [],
            0,
        ),
        "array3d": DatasetFixture(
            other(np.zeros((2, 2, 2), dtype="float32"), Array3D(shape=(2, 2, 2), dtype="float32")),
            {"_type": "Array3D", "shape": (2, 2, 2), "dtype": "float32"},
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            [],
            0,
        ),
        "array4d": DatasetFixture(
            other(np.zeros((2, 2, 2, 2), dtype="float32"), Array4D(shape=(2, 2, 2, 2), dtype="float32")),
            {"_type": "Array4D", "shape": (2, 2, 2, 2), "dtype": "float32"},
            [
                [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            ],
            [],
            0,
        ),
        "array5d": DatasetFixture(
            other(np.zeros((2, 2, 2, 2, 2), dtype="float32"), Array5D(shape=(2, 2, 2, 2, 2), dtype="float32")),
            {"_type": "Array5D", "shape": (2, 2, 2, 2, 2), "dtype": "float32"},
            [
                [
                    [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                ],
                [
                    [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                ],
            ],
            [],
            0,
        ),
        "translation": DatasetFixture(
            other({"en": "the cat", "fr": "le chat"}, Translation(languages=["en", "fr"])),
            {"_type": "Translation", "languages": ["en", "fr"]},
            {"en": "the cat", "fr": "le chat"},
            [],
            0,
        ),
        "translation_variable_languages": DatasetFixture(
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
                "language": ["en", "fr", "fr"],
                "translation": ["the cat", "la chatte", "le chat"],
            },
            [],
            0,
        ),
        "audio": DatasetFixture(
            other(
                {"array": [0.1, 0.2, 0.3], "sampling_rate": DEFAULT_SAMPLING_RATE},
                Audio(sampling_rate=DEFAULT_SAMPLING_RATE),
            ),
            {
                "_type": "Audio",
                "sampling_rate": DEFAULT_SAMPLING_RATE,  # <- why do we have a sampling rate here? (see below for contrast)
            },
            [
                {
                    "src": f"{ASSETS_BASE_URL}/audio/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/audio.wav",
                    "type": "audio/wav",
                }
            ],
            [AssetUrlPath(feature_type="Audio", path=[DEFAULT_COLUMN_NAME, 0])],
            1,
        ),
        "audio_ogg": DatasetFixture(
            other(
                str(Path(__file__).resolve().parent / "data" / "test_audio_vorbis.ogg"),
                Audio(sampling_rate=DEFAULT_SAMPLING_RATE),
            ),
            {
                "_type": "Audio",
                "sampling_rate": DEFAULT_SAMPLING_RATE,
            },
            [
                {
                    "src": f"{ASSETS_BASE_URL}/audio_ogg/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/audio.wav",
                    "type": "audio/wav",
                }
            ],
            [AssetUrlPath(feature_type="Audio", path=[DEFAULT_COLUMN_NAME, 0])],
            1,
        ),
        "image": DatasetFixture(
            other(
                str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
                Image(),
            ),
            {"_type": "Image"},
            {
                "src": f"{ASSETS_BASE_URL}/image/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/image.jpg",
                "height": 480,
                "width": 640,
            },
            [
                AssetUrlPath(
                    feature_type="Image",
                    path=[
                        DEFAULT_COLUMN_NAME,
                    ],
                )
            ],
            1,
        ),
        "images_list": DatasetFixture(
            other(
                [
                    str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
                    str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
                    None,
                    # ^ image lists can include nulls
                ],
                [Image()],
            ),
            [
                {
                    "_type": "Image",
                }
            ],
            [
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
                None,
            ],
            [AssetUrlPath(feature_type="Image", path=[DEFAULT_COLUMN_NAME, 0])],
            2,
        ),
        "audios_list": DatasetFixture(
            other(
                [
                    {"array": [0.1, 0.2, 0.3], "sampling_rate": DEFAULT_SAMPLING_RATE},
                    {"array": [0.1, 0.2, 0.3], "sampling_rate": DEFAULT_SAMPLING_RATE},
                    None,
                    # ^ audio lists can include nulls
                ],
                [Audio()],
            ),
            [
                {
                    "_type": "Audio",  # <- why don't we have a sampling rate here?
                }
            ],
            [
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
                None,
            ],
            [AssetUrlPath(feature_type="Audio", path=[DEFAULT_COLUMN_NAME, 0, 0])],
            2,
        ),
        "images_sequence": DatasetFixture(
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
            [
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
            ],
            [AssetUrlPath(feature_type="Image", path=[DEFAULT_COLUMN_NAME, 0])],
            2,
        ),
        "images_sequence_dict": DatasetFixture(
            other(
                {
                    "images": [
                        str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
                        str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
                    ]
                },
                Sequence(feature={"images": Image()}),
            ),
            {
                "_type": "Sequence",
                "feature": {
                    "images": {
                        "_type": "Image",
                    }
                },
            },
            {
                "images": [
                    {
                        "src": f"{ASSETS_BASE_URL}/images_sequence_dict/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/image-1d9603ef.jpg",
                        "height": 480,
                        "width": 640,
                    },
                    {
                        "src": f"{ASSETS_BASE_URL}/images_sequence_dict/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}/{DEFAULT_ROW_IDX}/col/image-1d9803f0.jpg",
                        "height": 480,
                        "width": 640,
                    },
                ]
            },
            [AssetUrlPath(feature_type="Image", path=[DEFAULT_COLUMN_NAME, "images", 0])],
            2,
        ),
        "audios_sequence": DatasetFixture(
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
            [
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
            ],
            [AssetUrlPath(feature_type="Audio", path=[DEFAULT_COLUMN_NAME, 0, 0])],
            2,
        ),
        "dict_of_audios_and_images": DatasetFixture(
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
            },
            [
                AssetUrlPath(feature_type="Image", path=[DEFAULT_COLUMN_NAME, "b", 0]),
                AssetUrlPath(feature_type="Audio", path=[DEFAULT_COLUMN_NAME, "c", "ca", 0, 0]),
            ],
            4,
        ),
        "sequence_of_dicts": DatasetFixture(
            other([{"a": {"b": 0}}, {"a": {"b": 1}}], Sequence(feature={"a": {"b": Value(dtype="int64")}})),
            {
                "_type": "Sequence",
                "feature": {
                    "a": {"b": {"_type": "Value", "dtype": "int64"}},
                },
            },
            {"a": [{"b": 0}, {"b": 1}]},
            [],
            0,
        ),
        "none_value": DatasetFixture(
            other({"a": None}, {"a": Value(dtype="int64")}),
            {"a": {"_type": "Value", "dtype": "int64"}},
            {"a": None},
            [],
            0,
        ),
        "big": DatasetFixture(
            Dataset.from_pandas(
                pd.DataFrame(
                    ["a" * 1_234 for _ in range(4_567)],
                    dtype=pd.StringDtype(storage="python"),
                )
            ),
            {"_type": "Value", "dtype": "string"},
            "a" * 1_234,
            [],
            0,
        ),
    }
