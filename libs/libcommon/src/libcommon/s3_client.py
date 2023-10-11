# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.
import fsspec
from typing import Any

class S3ClientInitializeError(Exception):
    pass


class S3Client:
    """
    A resource that represents a connection to S3.

    Args:
        protocol (:obj:`str`): The fsspec protocol (supported s3 or file)
        root (:obj:`str`): The storage root path
    """

    _fs: Any
    _storage_root: str

    def __init__(self, protocol: str, root: str, **kwargs: Any) -> None:
        try:
            self._storage_root = root
            if(protocol == "s3"):
                self._fs = fsspec.filesystem(protocol, **kwargs)
            else:
                self._fs = fsspec.filesystem(protocol, auto_mkdir=True)
            print(type(self._fs))
        except Exception as e:
            self._fs = None

    def is_available(self) -> bool:
        return self._fs is not None

    def exists(self, object_key: str) -> bool:
        if not self.is_available():
            raise S3ClientInitializeError()
        return self._fs.exists(f"{self._storage_root}/{object_key}")
