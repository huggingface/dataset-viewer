# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import datetime
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from datasets import Audio, Dataset, Image, Value
from libcommon.resources import StrPath

from datasets_based.config import FirstRowsConfig
from datasets_based.features import get_cell_value

# we need to know the correspondence between the feature type and the cell value, in order to:
# - document the API
# - implement the client on the Hub (dataset viewer)


# see https://github.com/huggingface/datasets/blob/a5192964dc4b76ee5c03593c11ee56f29bbd688d/...
#     src/datasets/features/features.py#L1469
# ``FieldType`` can be one of the following:
# - a :class:`datasets.Value` feature specifies a single typed value, e.g. ``int64`` or ``string``
@pytest.mark.parametrize(
    "dataset_type,output_value,output_dtype",
    [
        ("null", None, "null"),
        ("bool", False, "bool"),
        ("int8", -7, "int8"),
        ("int16", -7, "int16"),
        ("int32", -7, "int32"),
        ("int64", -7, "int64"),
        ("uint8", 7, "uint8"),
        ("uint16", 7, "uint16"),
        ("uint32", 7, "uint32"),
        ("uint64", 7, "uint64"),
        ("float16", np.float16(-3.14), "float16"),
        # (alias float)
        ("float32", np.float32(-3.14), "float32"),
        # (alias double)
        ("float64", -3.14, "float64"),
        ("time", datetime.time(1, 1, 1), "time64[us]"),
        ("timestamp_1", datetime.datetime(2020, 1, 1, 0, 0), "timestamp[ns]"),
        ("timestamp_2", datetime.datetime(2017, 12, 16, 3, 2, 35, 500000), "timestamp[ns]"),
        ("timestamp_3", datetime.datetime(2017, 12, 16, 3, 2, 35, 500000), "timestamp[ns]"),
        (
            "timestamp_tz",
            datetime.datetime(2020, 1, 1, 0, 0, tzinfo=ZoneInfo("US/Pacific")),
            "timestamp[ns, tz=US/Pacific]",
        ),
        ("string", "a string", "string"),
    ],
)
def test_value(
    dataset_type: str,
    output_value: Any,
    output_dtype: str,
    datasets: Mapping[str, Dataset],
    first_rows_config: FirstRowsConfig,
    assets_directory: StrPath,
) -> None:
    dataset = datasets[dataset_type]
    feature = dataset.features["col"]
    assert feature._type == "Value"
    assert feature.dtype == output_dtype
    value = get_cell_value(
        dataset="dataset",
        config="config",
        split="split",
        row_idx=7,
        cell=dataset[0]["col"],
        featureName="col",
        fieldType=feature,
        assets_base_url=first_rows_config.assets.base_url,
        assets_directory=assets_directory,
    )
    assert value == output_value


@pytest.mark.parametrize(
    "dataset_type,output_value,output_type",
    [
        # - a :class:`datasets.ClassLabel` feature specifies a field with a predefined set of classes
        #   which can have labels associated to them and will be stored as integers in the dataset
        ("class_label", 1, "ClassLabel"),
        # - a python :obj:`dict` which specifies that the field is a nested field containing a mapping of sub-fields
        #   to sub-fields features. It's possible to have nested fields of nested fields in an arbitrary manner
        ("dict", {"a": 0}, {"a": Value(dtype="int64", id=None)}),
        # - a python :obj:`list` or a :class:`datasets.Sequence` specifies that the field contains a list of objects.
        #    The python :obj:`list` or :class:`datasets.Sequence` should be provided with a single sub-feature as an
        #    example of the feature type hosted in this list
        #   <Tip>
        #   A :class:`datasets.Sequence` with a internal dictionary feature will be automatically converted into a
        #   dictionary of lists. This behavior is implemented to have a compatilbity layer with the TensorFlow Datasets
        #   library but may be un-wanted in some cases. If you don't want this behavior, you can use a python
        #   :obj:`list` instead of the :class:`datasets.Sequence`.
        #   </Tip>
        ("list", [{"a": 0}], [{"a": Value(dtype="int64", id=None)}]),
        ("sequence_simple", [0], "Sequence"),
        ("sequence", {"a": [0]}, "Sequence"),
        # - a :class:`Array2D`, :class:`Array3D`, :class:`Array4D` or :class:`Array5D` feature for multidimensional
        #   arrays
        ("array2d", [[0.0, 0.0], [0.0, 0.0]], "Array2D"),
        ("array3d", [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]], "Array3D"),
        (
            "array4d",
            [
                [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            ],
            "Array4D",
        ),
        (
            "array5d",
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
            "Array5D",
        ),
        # - an :class:`Audio` feature to store the absolute path to an audio file or a dictionary with the relative
        #   path to an audio file ("path" key) and its bytes content ("bytes" key). This feature extracts the audio
        #   data.
        (
            "audio",
            [
                {
                    "src": "http://localhost/assets/dataset/--/config/split/7/col/audio.mp3",
                    "type": "audio/mpeg",
                },
                {
                    "src": "http://localhost/assets/dataset/--/config/split/7/col/audio.wav",
                    "type": "audio/wav",
                },
            ],
            "Audio",
        ),
        # - an :class:`Image` feature to store the absolute path to an image file, an :obj:`np.ndarray` object, a
        #   :obj:`PIL.Image.Image` object or a dictionary with the relative path to an image file ("path" key) and
        #   its bytes content ("bytes" key). This feature extracts the image data.
        (
            "image",
            {
                "src": "http://localhost/assets/dataset/--/config/split/7/col/image.jpg",
                "height": 480,
                "width": 640,
            },
            "Image",
        ),
        # - :class:`datasets.Translation` and :class:`datasets.TranslationVariableLanguages`, the two features
        #   specific to Machine Translation
        ("translation", {"en": "the cat", "fr": "le chat"}, "Translation"),
        (
            "translation_variable_languages",
            {"language": ["en", "fr", "fr"], "translation": ["the cat", "la chatte", "le chat"]},
            "TranslationVariableLanguages",
        ),
        # special cases
        (
            "images_list",
            [
                {
                    "src": "http://localhost/assets/dataset/--/config/split/7/col/image-1d100e9.jpg",
                    "height": 480,
                    "width": 640,
                },
                {
                    "src": "http://localhost/assets/dataset/--/config/split/7/col/image-1d300ea.jpg",
                    "height": 480,
                    "width": 640,
                },
            ],
            [Image(decode=True, id=None)],
        ),
        (
            "audios_list",
            [
                [
                    {
                        "src": "http://localhost/assets/dataset/--/config/split/7/col/audio-1d100e9.mp3",
                        "type": "audio/mpeg",
                    },
                    {
                        "src": "http://localhost/assets/dataset/--/config/split/7/col/audio-1d100e9.wav",
                        "type": "audio/wav",
                    },
                ],
                [
                    {
                        "src": "http://localhost/assets/dataset/--/config/split/7/col/audio-1d300ea.mp3",
                        "type": "audio/mpeg",
                    },
                    {
                        "src": "http://localhost/assets/dataset/--/config/split/7/col/audio-1d300ea.wav",
                        "type": "audio/wav",
                    },
                ],
            ],
            [Audio()],
        ),
        (
            "images_sequence",
            [
                {
                    "src": "http://localhost/assets/dataset/--/config/split/7/col/image-1d100e9.jpg",
                    "height": 480,
                    "width": 640,
                },
                {
                    "src": "http://localhost/assets/dataset/--/config/split/7/col/image-1d300ea.jpg",
                    "height": 480,
                    "width": 640,
                },
            ],
            "Sequence",
        ),
        (
            "audios_sequence",
            [
                [
                    {
                        "src": "http://localhost/assets/dataset/--/config/split/7/col/audio-1d100e9.mp3",
                        "type": "audio/mpeg",
                    },
                    {
                        "src": "http://localhost/assets/dataset/--/config/split/7/col/audio-1d100e9.wav",
                        "type": "audio/wav",
                    },
                ],
                [
                    {
                        "src": "http://localhost/assets/dataset/--/config/split/7/col/audio-1d300ea.mp3",
                        "type": "audio/mpeg",
                    },
                    {
                        "src": "http://localhost/assets/dataset/--/config/split/7/col/audio-1d300ea.wav",
                        "type": "audio/wav",
                    },
                ],
            ],
            "Sequence",
        ),
        (
            "dict_of_audios_and_images",
            {
                "a": 0,
                "b": [
                    {
                        "src": "http://localhost/assets/dataset/--/config/split/7/col/image-89101db.jpg",
                        "height": 480,
                        "width": 640,
                    },
                    {
                        "src": "http://localhost/assets/dataset/--/config/split/7/col/image-89301dc.jpg",
                        "height": 480,
                        "width": 640,
                    },
                ],
                "c": {
                    "ca": [
                        [
                            {
                                "src": "http://localhost/assets/dataset/--/config/split/7/col/audio-18360330.mp3",
                                "type": "audio/mpeg",
                            },
                            {
                                "src": "http://localhost/assets/dataset/--/config/split/7/col/audio-18360330.wav",
                                "type": "audio/wav",
                            },
                        ],
                        [
                            {
                                "src": "http://localhost/assets/dataset/--/config/split/7/col/audio-18380331.mp3",
                                "type": "audio/mpeg",
                            },
                            {
                                "src": "http://localhost/assets/dataset/--/config/split/7/col/audio-18380331.wav",
                                "type": "audio/wav",
                            },
                        ],
                    ]
                },
            },
            {"a": Value(dtype="int64"), "b": [Image(decode=True, id=None)], "c": {"ca": [Audio()]}},
        ),
        ("sequence_of_dicts", {"a": [{"b": 0}, {"b": 1}]}, "Sequence"),
        ("none_value", {"a": None}, {"a": Value(dtype="int64", id=None)}),
    ],
)
def test_others(
    dataset_type: str,
    output_value: Any,
    output_type: Any,
    datasets: Mapping[str, Dataset],
    first_rows_config: FirstRowsConfig,
    assets_directory: StrPath,
) -> None:
    dataset = datasets[dataset_type]
    feature = dataset.features["col"]
    if type(output_type) in [list, dict]:
        assert feature == output_type
    else:
        assert feature._type == output_type
    value = get_cell_value(
        dataset="dataset",
        config="config",
        split="split",
        row_idx=7,
        cell=dataset[0]["col"],
        featureName="col",
        fieldType=feature,
        assets_base_url=first_rows_config.assets.base_url,
        assets_directory=assets_directory,
    )
    assert value == output_value
