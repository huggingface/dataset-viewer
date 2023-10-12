# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.
import fsspec
from typing import Any

class StorageClientInitializeError(Exception):
    pass


class StorageClient:
    """
    A resource that represents a connection to a storage client.

    Args:
        protocol (:obj:`str`): The fsspec protocol (supported s3 or file)
        root (:obj:`str`): The storage root path
    """

    _fs: Any
    _storage_root: str
    _folder: str

    def __init__(self, protocol: str, root: str, folder:str, **kwargs: Any) -> None:
        self._storage_root = root
        self._folder = folder
        if(protocol == "s3"):
            self._fs = fsspec.filesystem(protocol, **kwargs)
        elif(protocol == "file"):
            self._fs = fsspec.filesystem(protocol, auto_mkdir=True)
        else:
            raise StorageClientInitializeError("unsupported protocol")

    def exists(self, object_key: str) -> bool:
        return self._fs.exists(f"{self._storage_root}/{self._folder}/{object_key}")
