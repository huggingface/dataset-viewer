# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.
from typing import Optional

import boto3

S3_RESOURCE = "s3"


class S3Client:
    """
    A resource that represents a connection to S3.

    The method is_available() allows to check if the resource is available. It's not called automatically.

    Args:
        region_name (:obj:`str`): The AWS region
        aws_access_key_id (:obj:`str`): The AWS access key id,
        aws_secret_access_key (:obj:`str`): The AWS secret access key,
    """

    _client: boto3.session.Session.client = None

    def __init__(
        self, region_name: str, aws_access_key_id: Optional[str], aws_secret_access_key: Optional[str]
    ) -> None:
        self._client = boto3.client(
            S3_RESOURCE,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    def is_available(self) -> bool:
        """Check if the client has been initialized."""
        return self._client is not None

    def exists_in_bucket(self, bucket: str, object_key: str) -> bool:
        if not self.is_available():
            raise Exception("S3 Resource is not initialized")
        try:
            self._client.head_object(Bucket=bucket, Key=object_key)
            return True
        except Exception:
            return False

    def upload_to_bucket(self, file_path: str, bucket: str, object_key: str) -> None:
        self._client.upload_file(file_path, bucket, object_key)
