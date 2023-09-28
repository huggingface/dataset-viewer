# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from tempfile import NamedTemporaryFile

import boto3
from moto import mock_s3

from libcommon.s3_client import S3Client


def test_is_available() -> None:
    region = "us-east-1"
    bucket_name = "bucket"

    # validate client not available if configuration is wrong
    s3_client = S3Client(region_name="non-existent-region", bucket_name=bucket_name)
    assert not s3_client.is_available()

    # validate client is available if configuration is correct
    with mock_s3():
        conn = boto3.resource("s3", region_name=region)
        conn.create_bucket(Bucket=bucket_name)
        s3_client = S3Client(region_name=region, bucket_name=bucket_name)
        assert s3_client.is_available()


def test_upload_file_and_verify_if_exists() -> None:
    with mock_s3():
        region = "us-east-1"
        bucket_name = "bucket"
        conn = boto3.resource("s3", region_name=region)
        conn.create_bucket(Bucket=bucket_name)
        s3_client = S3Client(region_name=region, bucket_name=bucket_name)
        assert s3_client.is_available()

        # validate object does not exist in bucket
        assert not s3_client.exists(object_key="non_existent")

        # validate object exists in bucket after uploading
        with NamedTemporaryFile() as tmp:
            s3_client.upload(file_path=tmp.name, object_key="sample")
            assert s3_client.exists(object_key="sample")
