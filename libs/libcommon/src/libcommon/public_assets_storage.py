# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.
from dataclasses import dataclass

from libcommon.storage_client import StorageClient


@dataclass
class PublicAssetsStorage:
    overwrite: bool
    storage_client: StorageClient
