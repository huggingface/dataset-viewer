# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.
import logging
from typing import Optional

import boto3

S3_RESOURCE = "s3"


class S3ClientInitializeError(Exception):
    pass


class S3Client:
    """
    A resource that represents a connection to S3.

    Args:
        region_name (:obj:`str`): The AWS region
        aws_access_key_id (:obj:`str`): The AWS access key id,
        aws_secret_access_key (:obj:`str`): The AWS secret access key,
    """

    _client: boto3.session.Session.client = None
    _bucket_name: str

    def __init__(
        self,
        region_name: str,
        bucket_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> None:
        try:
            self._bucket_name = bucket_name
            self._client = boto3.client(
                S3_RESOURCE,
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
            logging.debug("verify bucket exists and client has permissions")
            self._client.head_bucket(Bucket=self._bucket_name)
            logging.info("s3 client was configured/initialized correctly")
        except Exception as e:
            logging.error(f"s3 client was not configured/initialized correctly {e}")
            self._client = None

    def is_available(self) -> bool:
        return self._client is not None

    def exists(self, object_key: str) -> bool:
        logging.debug(f"validate object {object_key=} exists on {self._bucket_name}")
        if not self.is_available():
            raise S3ClientInitializeError()
        try:
            self._client.head_object(Bucket=self._bucket_name, Key=object_key)
            return True
        except Exception:
            return False

    def upload(self, file_path: str, object_key: str) -> None:
        logging.debug(f"upload object {object_key=} to {self._bucket_name}")
        if not self.is_available():
            raise S3ClientInitializeError()
        self._client.upload_file(file_path, self._bucket_name, object_key)
