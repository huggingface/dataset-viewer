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

    def __init__(
        self, region_name: str, aws_access_key_id: Optional[str] = None, aws_secret_access_key: Optional[str] = None
    ) -> None:
        try:
            logging.debug(
                f"init s3 client with {region_name=} aws_access_key_id exists: {aws_access_key_id is not None} -"
                f" aws_secret_access_key exists: {aws_secret_access_key is not None} "
            )
            self._client = boto3.client(
                S3_RESOURCE,
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
            # no needed, just helps verify client has been configured correctly
            logging.debug("trying to list buckets to verify configuration")
            self._client.list_buckets()
            logging.info("client was configured/initialized correctly")
        except Exception as e:
            logging.error(f"client was not configured/initialized correctly {e}")
            self._client = None

    def is_available(self) -> bool:
        return self._client is not None

    def exists_in_bucket(self, bucket: str, object_key: str) -> bool:
        logging.debug(f"validate object {object_key=} exists on {bucket=}")
        if not self.is_available():
            raise S3ClientInitializeError()
        try:
            self._client.head_object(Bucket=bucket, Key=object_key)
            return True
        except Exception:
            return False

    def upload_to_bucket(self, file_path: str, bucket: str, object_key: str) -> None:
        logging.debug(f"upload object {object_key=} to {bucket=}")
        if not self.is_available():
            raise S3ClientInitializeError()
        self._client.upload_file(file_path, bucket, object_key)
