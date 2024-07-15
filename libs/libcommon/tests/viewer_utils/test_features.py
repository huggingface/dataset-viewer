# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import boto3
import pytest
from aiobotocore.response import StreamingBody
from datasets import Audio, Features, Image, Value
from moto import mock_s3
from urllib3._collections import HTTPHeaderDict  # type: ignore

from libcommon.config import S3Config
from libcommon.storage_client import StorageClient
from libcommon.url_preparator import URLPreparator
from libcommon.viewer_utils.features import (
    get_cell_value,
    get_supported_unsupported_columns,
    infer_audio_file_extension,
    to_features_list,
)

from ..constants import (
    ASSETS_BASE_URL,
    DATASETS_NAMES,
    DEFAULT_COLUMN_NAME,
    DEFAULT_CONFIG,
    DEFAULT_REVISION,
    DEFAULT_ROW_IDX,
    DEFAULT_SPLIT,
)
from ..types import DatasetFixture


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


@pytest.mark.parametrize("decoded", [True, False])
@pytest.mark.parametrize("dataset_name", DATASETS_NAMES)
def test_get_cell_value_value(
    storage_client_with_url_preparator: StorageClient,
    datasets_fixtures: Mapping[str, DatasetFixture],
    dataset_name: str,
    decoded: bool,
) -> None:
    dataset_fixture = datasets_fixtures[dataset_name]
    dataset = dataset_fixture.dataset
    feature = dataset.features[DEFAULT_COLUMN_NAME]
    cell = (
        dataset[DEFAULT_ROW_IDX][DEFAULT_COLUMN_NAME]
        if decoded
        else dataset.with_format("arrow")[DEFAULT_ROW_IDX].to_pydict()[DEFAULT_COLUMN_NAME][DEFAULT_ROW_IDX]
    )
    expected_cell = dataset_fixture.expected_cell
    value = get_cell_value(
        dataset=dataset_name,
        revision=DEFAULT_REVISION,
        config=DEFAULT_CONFIG,
        split=DEFAULT_SPLIT,
        row_idx=DEFAULT_ROW_IDX,
        cell=cell,
        featureName=DEFAULT_COLUMN_NAME,
        fieldType=feature,
        storage_client=storage_client_with_url_preparator,
    )
    assert value == expected_cell
    assert_output_has_valid_files(expected_cell, storage_client=storage_client_with_url_preparator)


to_features_list


@pytest.mark.parametrize("dataset_name", DATASETS_NAMES)
def test_to_features_list(
    datasets_fixtures: Mapping[str, DatasetFixture],
    dataset_name: str,
) -> None:
    datasets_fixture = datasets_fixtures[dataset_name]
    dataset = datasets_fixture.dataset
    value = to_features_list(dataset.features)
    assert len(value) == 1
    first_feature = value[0]
    assert first_feature["feature_idx"] == 0
    assert first_feature["name"] == DEFAULT_COLUMN_NAME
    assert first_feature["type"] == datasets_fixture.expected_feature_type


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


# specific test created for https://github.com/huggingface/dataset-viewer/issues/2045
# which is reproduced only when using s3 for fsspec
def test_ogg_audio_with_s3(
    datasets_fixtures: Mapping[str, DatasetFixture],
) -> None:
    dataset_name = "audio_ogg"
    dataset_fixture = datasets_fixtures[dataset_name]
    dataset = dataset_fixture.dataset
    feature = dataset.features[DEFAULT_COLUMN_NAME]
    bucket_name = "bucket"
    with mock_s3():
        conn = boto3.resource("s3", region_name="us-east-1")
        conn.create_bucket(Bucket=bucket_name)

        # patch _validate to avoid calling self._fs.ls because of known issue in aiotbotocore
        # at https://github.com/aio-libs/aiobotocore/blob/master/aiobotocore/endpoint.py#L47
        with patch("libcommon.storage_client.StorageClient._validate", return_value=None):
            storage_client = StorageClient(
                protocol="s3",
                storage_root=f"{bucket_name}/not-important",
                base_url=ASSETS_BASE_URL,
                overwrite=True,
                s3_config=S3Config(
                    access_key_id="fake_access_key_id",
                    secret_access_key="fake_secret_access_key",
                    region_name="us-east-1",
                ),
                url_preparator=URLPreparator(url_signer=None),
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
                dataset=dataset_name,
                revision=DEFAULT_REVISION,
                config=DEFAULT_CONFIG,
                split=DEFAULT_SPLIT,
                row_idx=DEFAULT_ROW_IDX,
                cell=dataset[DEFAULT_ROW_IDX][DEFAULT_COLUMN_NAME],
                featureName=DEFAULT_COLUMN_NAME,
                fieldType=feature,
                storage_client=storage_client,
            )
            assert value == dataset_fixture.expected_cell


@pytest.mark.parametrize(
    "audio_file_name,expected_audio_file_extension",
    [("test_audio_44100.wav", ".wav"), ("test_audio_16000.mp3", ".mp3")],
)
def test_infer_audio_file_extension(
    audio_file_name: str, expected_audio_file_extension: str, shared_datadir: Path
) -> None:
    audio_file_bytes = (shared_datadir / audio_file_name).read_bytes()
    audio_file_extension = infer_audio_file_extension(audio_file_bytes)
    assert audio_file_extension == expected_audio_file_extension
