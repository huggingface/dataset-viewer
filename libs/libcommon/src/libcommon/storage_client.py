# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.
import logging
from typing import Any, Optional

import fsspec

from libcommon.cloudfront import CloudFront
from libcommon.config import CloudFrontConfig, S3Config


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
        cloudfront_config (`CloudFrontConfig`, *optional*): The CloudFront configuration to generate signed urls
    """

    _fs: Any
    protocol: str
    storage_root: str
    base_url: str
    overwrite: bool
    cloudfront: Optional[CloudFront] = None

    def __init__(
        self,
        protocol: str,
        storage_root: str,
        base_url: str,
        overwrite: bool = False,
        s3_config: Optional[S3Config] = None,
        cloudfront_config: Optional[CloudFrontConfig] = None,
    ) -> None:
        logging.info(f"trying to initialize storage client with {protocol=} {storage_root=} {base_url=} {overwrite=}")
        self.storage_root = storage_root
        self.protocol = protocol
        self.base_url = base_url
        self.overwrite = overwrite
        if protocol == "s3":
            if not s3_config:
                raise StorageClientInitializeError("s3 config is required")
            self._fs = fsspec.filesystem(
                protocol,
                key=s3_config.access_key_id,
                secret=s3_config.secret_access_key,
                client_kwargs={"region_name": s3_config.region_name},
            )
            if cloudfront_config and cloudfront_config.key_pair_id and cloudfront_config.private_key:
                # ^ signed urls are enabled if the key pair id is passed in the configuration
                self.cloudfront = CloudFront(
                    key_pair_id=cloudfront_config.key_pair_id,
                    private_key=cloudfront_config.private_key,
                    expiration_seconds=cloudfront_config.expiration_seconds,
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
        if self.cloudfront:
            url = self.cloudfront.sign_url(url=url)
            logging.debug(f"signed url: {url}")
        return url

    def delete_dataset_directory(self, dataset: str) -> None:
        dataset_key = self.get_full_path(dataset)
        try:
            self._fs.rm(dataset_key, recursive=True)
            logging.info(f"Directory deleted: {dataset_key}")
        except Exception:
            logging.warning(f"Could not delete directory {dataset_key}")

    def __repr__(self) -> str:
        return f"StorageClient(protocol={self.protocol}, storage_root={self.storage_root}, base_url={self.base_url}, overwrite={self.overwrite}), cloudfront={self.cloudfront})"
