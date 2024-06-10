# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.
import logging
from typing import Optional, Union

import fsspec
from fsspec.implementations.local import LocalFileSystem
from s3fs import S3FileSystem  # type: ignore

from libcommon.config import S3Config, StorageProtocol
from libcommon.url_signer import URLSigner


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
        url_signer (`URLSigner`, *optional*): The url signer to use for signing urls
    """

    _fs: Union[LocalFileSystem, S3FileSystem]
    protocol: StorageProtocol
    storage_root: str
    base_url: str
    overwrite: bool
    url_signer: Optional[URLSigner] = None

    def __init__(
        self,
        protocol: StorageProtocol,
        storage_root: str,
        base_url: str,
        overwrite: bool = False,
        s3_config: Optional[S3Config] = None,
        url_signer: Optional[URLSigner] = None,
    ) -> None:
        logging.info(f"trying to initialize storage client with {protocol=} {storage_root=} {base_url=} {overwrite=}")
        self.storage_root = storage_root
        self.protocol = protocol
        self.base_url = base_url
        self.overwrite = overwrite
        self.url_signer = url_signer
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

    def get_url(self, path: str) -> str:
        return self.sign_url_if_available(self.get_unsigned_url(path))

    def get_unsigned_url(self, path: str) -> str:
        url = f"{self.base_url}/{path}"
        logging.debug(f"unsigned url: {url}")
        return url

    def sign_url_if_available(self, url: str) -> str:
        if self.url_signer:
            url = self.url_signer.sign_url(url=url)
            logging.debug(f"signed url: {url}")
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

    def __str__(self) -> str:
        return f"StorageClient(protocol={self.protocol}, storage_root={self.storage_root}, base_url={self.base_url}, overwrite={self.overwrite}, url_signer={self.url_signer})"
