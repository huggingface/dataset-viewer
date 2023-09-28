# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.
from dataclasses import dataclass

from libcommon.s3_client import S3Client
from libcommon.storage import StrPath


@dataclass
class DirectoryStorageOptions:
    assets_base_url: str
    assets_directory: StrPath
    overwrite: bool


@dataclass
class S3StorageOptions(DirectoryStorageOptions):
    s3_folder_name: str
    s3_client: S3Client
