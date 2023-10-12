# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from tempfile import NamedTemporaryFile

import boto3
from moto import mock_s3

from libcommon.storage_client import StorageClient


# def test_is_available() -> None:
#     region = "us-east-1"
#     bucket_name = "bucket"

#     # validate client not available if configuration is wrong
#     storage_client = StorageClient(region_name="non-existent-region", bucket_name=bucket_name)
#     assert not storage_client.is_available()

#     # validate client is available if configuration is correct
#     with mock_s3():
#         conn = boto3.resource("s3", region_name=region)
#         conn.create_bucket(Bucket=bucket_name)
#         storage_client = StorageClient(region_name=region, bucket_name=bucket_name)
#         assert storage_client.is_available()
