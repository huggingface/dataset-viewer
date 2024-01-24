# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import datetime
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch
from zoneinfo import ZoneInfo

import boto3
import numpy as np
import pytest
from aiobotocore.response import StreamingBody
from datasets import Audio, Dataset, Features, Image, Value
from moto import mock_s3
from urllib3.response import HTTPHeaderDict  # type: ignore

from libcommon.config import S3Config
from libcommon.storage_client import StorageClient
from libcommon.viewer_utils.features import (
    get_cell_value,
    get_supported_unsupported_columns,
    infer_audio_file_extension,
)

ASSETS_FOLDER = "assets"
ASSETS_BASE_URL = f"http://localhost/{ASSETS_FOLDER}"


@pytest.fixture
def storage_client(tmp_path: Path) -> StorageClient:
    return StorageClient(
        protocol="file", storage_root=str(tmp_path / ASSETS_FOLDER), base_url=ASSETS_BASE_URL, overwrite=False
    )


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
    storage_client: StorageClient,
    dataset_type: str,
    output_value: Any,
    output_dtype: str,
    datasets: Mapping[str, Dataset],
) -> None:
    dataset = datasets[dataset_type]
    feature = dataset.features["col"]
    assert feature._type == "Value"
    assert feature.dtype == output_dtype
    value = get_cell_value(
        dataset="dataset",
        revision="revision",
        config="config",
        split="split",
        row_idx=7,
        cell=dataset[0]["col"],
        featureName="col",
        fieldType=feature,
        storage_client=storage_client,
    )
    assert value == output_value


def assert_output_has_valid_files(value: Any, storage_client: StorageClient) -> None:
    if isinstance(value, list):
        for item in value:
            assert_output_has_valid_files(item, storage_client=storage_client)
    elif isinstance(value, dict):
        if "src" in value and isinstance(value["src"], str) and value["src"].startswith(storage_client.base_url):
            path = Path(
                storage_client.get_full_path(
                    value["src"][len(storage_client.base_url) + 1 :],  # noqa: E203
                )
            )
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0


ASSETS_BASE_URL_SPLIT = "http://localhost/assets/dataset/--/revision/--/config/split"


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
                    "src": f"{ASSETS_BASE_URL_SPLIT}/7/col/audio.wav",
                    "type": "audio/wav",
                }
            ],
            "Audio",
        ),
        (
            "audio_ogg",
            [
                {
                    "src": f"{ASSETS_BASE_URL_SPLIT}/7/col/audio.wav",
                    "type": "audio/wav",
                }
            ],
            "Audio",
        ),
        # - an :class:`Image` feature to store the absolute path to an image file, an :obj:`np.ndarray` object, a
        #   :obj:`PIL.Image.Image` object or a dictionary with the relative path to an image file ("path" key) and
        #   its bytes content ("bytes" key). This feature extracts the image data.
        (
            "image",
            {
                "src": f"{ASSETS_BASE_URL_SPLIT}/7/col/image.jpg",
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
                    "src": f"{ASSETS_BASE_URL_SPLIT}/7/col/image-1d100e9.jpg",
                    "height": 480,
                    "width": 640,
                },
                {
                    "src": f"{ASSETS_BASE_URL_SPLIT}/7/col/image-1d300ea.jpg",
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
                        "src": f"{ASSETS_BASE_URL_SPLIT}/7/col/audio-1d100e9.wav",
                        "type": "audio/wav",
                    },
                ],
                [
                    {
                        "src": f"{ASSETS_BASE_URL_SPLIT}/7/col/audio-1d300ea.wav",
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
                    "src": f"{ASSETS_BASE_URL_SPLIT}/7/col/image-1d100e9.jpg",
                    "height": 480,
                    "width": 640,
                },
                {
                    "src": f"{ASSETS_BASE_URL_SPLIT}/7/col/image-1d300ea.jpg",
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
                        "src": f"{ASSETS_BASE_URL_SPLIT}/7/col/audio-1d100e9.wav",
                        "type": "audio/wav",
                    },
                ],
                [
                    {
                        "src": f"{ASSETS_BASE_URL_SPLIT}/7/col/audio-1d300ea.wav",
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
                        "src": f"{ASSETS_BASE_URL_SPLIT}/7/col/image-89101db.jpg",
                        "height": 480,
                        "width": 640,
                    },
                    {
                        "src": f"{ASSETS_BASE_URL_SPLIT}/7/col/image-89301dc.jpg",
                        "height": 480,
                        "width": 640,
                    },
                ],
                "c": {
                    "ca": [
                        [
                            {
                                "src": f"{ASSETS_BASE_URL_SPLIT}/7/col/audio-18360330.wav",
                                "type": "audio/wav",
                            },
                        ],
                        [
                            {
                                "src": f"{ASSETS_BASE_URL_SPLIT}/7/col/audio-18380331.wav",
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
    storage_client: StorageClient,
) -> None:
    dataset = datasets[dataset_type]
    feature = dataset.features["col"]
    if type(output_type) in [list, dict]:
        assert feature == output_type
    else:
        assert feature._type == output_type
    # decoded
    value = get_cell_value(
        dataset="dataset",
        revision="revision",
        config="config",
        split="split",
        row_idx=7,
        cell=dataset[0]["col"],
        featureName="col",
        fieldType=feature,
        storage_client=storage_client,
    )
    assert value == output_value
    assert_output_has_valid_files(output_value, storage_client=storage_client)
    # encoded
    value = get_cell_value(
        dataset="dataset",
        revision="revision",
        config="config",
        split="split",
        row_idx=7,
        cell=dataset.with_format("arrow")[0].to_pydict()["col"][0],
        featureName="col",
        fieldType=feature,
        storage_client=storage_client,
    )
    assert value == output_value
    assert_output_has_valid_files(output_value, storage_client=storage_client)


def test_get_supported_unsupported_columns() -> None:
    features = Features(
        {
            "audio1": Audio(),
            "audio2": Audio(sampling_rate=16_000),
            "audio3": [Audio()],
            "image1": Image(),
            "image2": Image(decode=False),
            "image3": [Image()],
            "string": Value("string"),
            "binary": Value("binary"),
        }
    )
    unsupported_features = [Value("binary"), Audio()]
    supported_columns, unsupported_columns = get_supported_unsupported_columns(features, unsupported_features)
    assert supported_columns == ["image1", "image2", "image3", "string"]
    assert unsupported_columns == ["audio1", "audio2", "audio3", "binary"]


# specific test created for https://github.com/huggingface/datasets-server/issues/2045
# which is reproduced only when using s3 for fsspec
def test_ogg_audio_with_s3(
    datasets: Mapping[str, Dataset],
) -> None:
    dataset = datasets["audio_ogg"]
    feature = dataset.features["col"]
    bucket_name = "bucket"
    with mock_s3():
        conn = boto3.resource("s3", region_name="us-east-1")
        conn.create_bucket(Bucket=bucket_name)

        # patch _validate to avoid calling self._fs.ls because of known issue in aiotbotocore
        # at https://github.com/aio-libs/aiobotocore/blob/master/aiobotocore/endpoint.py#L47
        with patch("libcommon.storage_client.StorageClient._validate", return_value=None):
            storage_client = StorageClient(
                protocol="s3",
                storage_root=f"{bucket_name}/{ASSETS_FOLDER}",
                base_url=ASSETS_BASE_URL,
                overwrite=True,
                s3_config=S3Config(
                    access_key_id="fake_access_key_id",
                    secret_access_key="fake_secret_access_key",
                    region_name="us-east-1",
                ),
            )

        # patch aiobotocore.endpoint.convert_to_response_dict  because of known issue in aiotbotocore
        # at https://github.com/aio-libs/aiobotocore/blob/master/aiobotocore/endpoint.py#L47
        # see https://github.com/getmoto/moto/issues/6836 and https://github.com/aio-libs/aiobotocore/issues/755
        # copied from https://github.com/aio-libs/aiobotocore/blob/master/aiobotocore/endpoint.py#L23
        async def convert_to_response_dict(http_response, operation_model):  # type: ignore
            response_dict = {
                "headers": HTTPHeaderDict({}),
                "status_code": http_response.status_code,
                "context": {
                    "operation_name": operation_model.name,
                },
            }
            if response_dict["status_code"] >= 300:
                response_dict["body"] = await http_response.content
            elif operation_model.has_event_stream_output:
                response_dict["body"] = http_response.raw
            elif operation_model.has_streaming_output:
                length = response_dict["headers"].get("content-length")
                response_dict["body"] = StreamingBody(http_response.raw, length)
            else:
                response_dict["body"] = http_response.content
            return response_dict

        with patch("aiobotocore.endpoint.convert_to_response_dict", side_effect=convert_to_response_dict):
            value = get_cell_value(
                dataset="dataset",
                revision="revision",
                config="config",
                split="split",
                row_idx=7,
                cell=dataset[0]["col"],
                featureName="col",
                fieldType=feature,
                storage_client=storage_client,
            )
            audio_key = "dataset/--/revision/--/config/split/7/col/audio.wav"
            assert value == [
                {
                    "src": f"{ASSETS_BASE_URL}/{audio_key}",
                    "type": "audio/wav",
                },
            ]


@pytest.mark.parametrize(
    "audio_file_name, expected_audio_file_extension",
    [("test_audio_44100.wav", ".wav"), ("test_audio_16000.mp3", ".mp3")],
)
def test_infer_audio_file_extension(
    audio_file_name: str, expected_audio_file_extension: str, shared_datadir: Path
) -> None:
    audio_file_bytes = (shared_datadir / audio_file_name).read_bytes()
    audio_file_extension = infer_audio_file_extension(audio_file_bytes)
    assert audio_file_extension == expected_audio_file_extension
