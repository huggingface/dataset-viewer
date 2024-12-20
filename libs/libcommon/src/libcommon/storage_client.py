# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.
import logging
from typing import Optional, Union
from urllib import parse

import fsspec
from fsspec.implementations.local import LocalFileSystem
from s3fs import S3FileSystem  # type: ignore

from libcommon.config import S3Config, StorageProtocol
from libcommon.constants import DATASET_SEPARATOR
from libcommon.url_preparator import URLPreparator


class StorageClientInitializeError(Exception):
    pass


class StorageClient:
    """
    A resource that represents a connection to a storage client.

    Args:
        protocol (`str`): The fsspec protocol (supported: "file" or "s3")
        storage_root (`str`): The storage root path
        base_url (`str`): The base url for the publicly distributed assets
        overwrite (`bool`, *optional*, defaults to `False`): Whether to overwrite existing files
        s3_config (`S3Config`, *optional*): The S3 configuration to connect to the storage client. Only needed if the protocol is "s3"
        url_preparator (`URLPreparator`, *optional*): The urlpreparator to use for signing urls and replacing revision in url
    """

    _fs: Union[LocalFileSystem, S3FileSystem]
    protocol: StorageProtocol
    storage_root: str
    base_url: str
    overwrite: bool
    url_preparator: Optional[URLPreparator] = None

    def __init__(
        self,
        protocol: StorageProtocol,
        storage_root: str,
        base_url: str,
        overwrite: bool = False,
        s3_config: Optional[S3Config] = None,
        url_preparator: Optional[URLPreparator] = None,
    ) -> None:
        logging.info(f"trying to initialize storage client with {protocol=} {storage_root=} {base_url=} {overwrite=}")
        self.storage_root = storage_root
        self.protocol = protocol
        self.base_url = base_url
        self.overwrite = overwrite
        self.url_preparator = url_preparator
        if protocol == "s3":
            if not s3_config:
                raise StorageClientInitializeError("s3 config is required")
            self._fs = fsspec.filesystem(
                protocol,
                key=s3_config.access_key_id,
                secret=s3_config.secret_access_key,
                client_kwargs={"region_name": s3_config.region_name},
                max_paths=100,  # to avoid the DirCache from growing too much
            )
        elif protocol == "file":
            self._fs = fsspec.filesystem(protocol, auto_mkdir=True)
        else:
            raise StorageClientInitializeError("unsupported protocol")
        self._validate()

    def _validate(self) -> None:
        try:
            self._fs.ls(self.storage_root)
        except FileNotFoundError:
            self._fs.mkdir(self.storage_root)
        except Exception as e:
            raise StorageClientInitializeError("error when trying to initialize client", e)

    def get_full_path(self, path: str) -> str:
        return f"{self.storage_root}/{path}"

    def exists(self, path: str) -> bool:
        return bool(self._fs.exists(self.get_full_path(path)))

    def get_url(self, path: str, revision: str) -> str:
        return self.prepare_url(self.get_unprepared_url(path), revision=revision)

    def get_unprepared_url(self, path: str) -> str:
        # handle path in assets and hf/http urls
        url = path if "://" in path else f"{self.base_url}/{path}"
        logging.debug(f"unprepared url: {url}")
        return url

    def prepare_url(self, url: str, revision: str) -> str:
        if self.url_preparator:
            url = self.url_preparator.prepare_url(url=url, revision=revision)
        logging.debug(f"prepared url: {url}")
        return url

    def delete_dataset_directory(self, dataset: str) -> int:
        """
        Delete a dataset directory

        Args:
            dataset (`str`): the dataset

        Returns:
            int: The number of directories deleted (0 or 1)
        """
        dataset_key = self.get_full_path(dataset)
        try:
            self._fs.rm(dataset_key, recursive=True)
            logging.info(f"Directory deleted: {dataset_key}")
            return 1
        except FileNotFoundError:
            return 0
        except Exception:
            logging.warning(f"Could not delete directory {dataset_key}")
            return 0

    def update_revision_of_dataset_revision_directory(self, dataset: str, old_revision: str, new_revision: str) -> int:
        """
        Update the reivsion of a dataset directory

        Args:
            dataset (`str`): the dataset
            old_revision (`str`): the old revision
            new_revision (`str`): the new revision

        Returns:
            int: The number of directories updated (0 or 1)
        """
        # same pattern as in asset.py
        old_dataset_revision_key = self.get_full_path(f"{parse.quote(dataset)}/{DATASET_SEPARATOR}/{old_revision}")
        new_dataset_revision_key = self.get_full_path(f"{parse.quote(dataset)}/{DATASET_SEPARATOR}/{new_revision}")
        try:
            self._fs.mv(old_dataset_revision_key, new_dataset_revision_key, recursive=True)
            logging.info(
                f"Revision of the directory updated: {old_dataset_revision_key} -> {new_dataset_revision_key}"
            )
            return 1
        except Exception:
            logging.warning(
                f"Could not update the revision of directory {old_dataset_revision_key} to {new_dataset_revision_key}"
            )
            return 0

    @staticmethod
    def generate_object_path(
        dataset: str, revision: str, config: str, split: str, row_idx: int, column: str, filename: str
    ) -> str:
        return f"{parse.quote(dataset)}/{DATASET_SEPARATOR}/{revision}/{DATASET_SEPARATOR}/{parse.quote(config)}/{parse.quote(split)}/{str(row_idx)}/{parse.quote(column)}/{filename}"

    def __str__(self) -> str:
        return f"StorageClient(protocol={self.protocol}, storage_root={self.storage_root}, base_url={self.base_url}, overwrite={self.overwrite}, url_preparator={self.url_preparator})"
