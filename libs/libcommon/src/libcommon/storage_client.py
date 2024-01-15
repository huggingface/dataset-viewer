# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.
import logging
from typing import Any

import fsspec


class StorageClientInitializeError(Exception):
    pass


class StorageClient:
    """
    A resource that represents a connection to a storage client.

    Args:
        protocol (:obj:`str`): The fsspec protocol (supported s3 or file)
        storage_root (:obj:`str`): The storage root path
        base_url (:obj:`str`): The base url for the publicly distributed assets
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`): Whether to overwrite existing files
    """

    _fs: Any
    protocol: str
    storage_root: str
    base_url: str
    overwrite: bool

    def __init__(
        self, protocol: str, storage_root: str, base_url: str, overwrite: bool = False, **kwargs: Any
    ) -> None:
        logging.info(f"trying to initialize storage client with {protocol=} {storage_root=}")
        self.storage_root = storage_root
        self.protocol = protocol
        self.base_url = base_url
        self.overwrite = overwrite
        if protocol == "s3":
            self._fs = fsspec.filesystem(protocol, **kwargs)
        elif protocol == "file":
            self._fs = fsspec.filesystem(protocol, auto_mkdir=True)
        else:
            raise StorageClientInitializeError("unsupported protocol")
        self._validate()

    def _validate(self) -> None:
        try:
            self._check_or_create(self.storage_root)
        except Exception as e:
            raise StorageClientInitializeError("error when trying to initialize client", e)

    def _check_or_create(self, path: str) -> None:
        try:
            self._fs.ls(path)
        except FileNotFoundError:
            self._fs.mkdir(path)

    def get_full_path(self, path: str) -> str:
        return f"{self.storage_root}/{path}"

    def exists(self, path: str) -> bool:
        return bool(self._fs.exists(self.get_full_path(path)))

    def get_url(self, path: str) -> str:
        return f"{self.base_url}/{path}"

    def delete_dataset_directory(self, dataset: str) -> None:
        dataset_key = self.get_full_path(dataset)
        try:
            self._fs.rm(dataset_key, recursive=True)
            logging.info(f"Directory deleted: {dataset_key}")
        except Exception:
            logging.warning(f"Could not delete directory {dataset_key}")

    def __repr__(self) -> str:
        return f"StorageClient(protocol={self.protocol}, storage_root={self.storage_root}, base_url={self.base_url}, overwrite={self.overwrite})"
