# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.


from datetime import datetime

import httpx
import pytest

from libcommon.config import AssetsConfig, CloudFrontConfig, S3Config
from libcommon.storage_client import StorageClient

BUCKET = "hf-datasets-server-statics-test"


def test_s3_access_key_id() -> None:
    s3_config = S3Config.from_env()
    assert s3_config.access_key_id is not None, len(s3_config.access_key_id)


def test_s3_secret_access_key() -> None:
    s3_config = S3Config.from_env()
    assert s3_config.secret_access_key is not None, len(s3_config.secret_access_key)


def test_cloudfront_key_pair_id() -> None:
    cloudfront_config = CloudFrontConfig.from_env()
    assert cloudfront_config.key_pair_id is not None, len(cloudfront_config.key_pair_id)


def test_cloudfront_private_key() -> None:
    cloudfront_config = CloudFrontConfig.from_env()
    assert cloudfront_config.private_key is not None, len(cloudfront_config.private_key)


def test_real_cloudfront() -> None:
    # this test is not mocked, and will fail if the credentials are not valid
    # it requires the following environment variables to be set:
    # - S3_ACCESS_KEY_ID
    # - S3_SECRET_ACCESS_KEY
    # - CLOUDFRONT_KEY_PAIR_ID
    # - CLOUDFRONT_PRIVATE_KEY
    # To run it locally, set the environment variables in .env and run:
    #     set -a && source .env && set +a && make test
    cloudfront_config = CloudFrontConfig.from_env()
    s3_config = S3Config.from_env()
    assets_config = AssetsConfig(
        base_url="https://datasets-server-test.us.dev.moon.huggingface.tech/assets",
        # ^ assets/ is hardcoded in cloudfront configuration
        storage_protocol="s3",
        storage_root=f"{BUCKET}/assets",
    )
    if (
        (s3_config.access_key_id is None)
        or (s3_config.secret_access_key is None)
        or (cloudfront_config.key_pair_id is None)
        or (cloudfront_config.private_key is None)
    ):
        pytest.skip("the S3 and/or CloudFront credentials are not set in environment variables, so we skip the test")

    storage_client = StorageClient(
        protocol=assets_config.storage_protocol,
        storage_root=assets_config.storage_root,
        base_url=assets_config.base_url,
        overwrite=True,
        cloudfront_config=cloudfront_config,
        key=s3_config.access_key_id,
        secret=s3_config.secret_access_key,
        client_kwargs={"region_name": s3_config.region_name},
    )
    DATASET = datetime.now().strftime("%Y%m%d-%H%M%S")
    # ^ we could delete them, or simply set a TTL in the bucket
    path = f"{DATASET}/path/to/a/file.txt"

    # the root exists
    storage_client.exists("/")
    # the dataset directory does not exist
    assert not storage_client.exists(DATASET)
    # the file does not exist
    assert not storage_client.exists(path)

    # can write a file
    with storage_client._fs.open(storage_client.get_full_path(path), "wt") as f:
        f.write("hello world")

    # the dataset directory exists
    assert storage_client.exists(DATASET)
    # the file exists
    assert storage_client.exists(path)

    # can read the file
    with storage_client._fs.open(storage_client.get_full_path(path), "rt") as f:
        assert f.read() == "hello world"

    # cannot access the file through the normal url
    unsigned_url = storage_client.get_unsigned_url(path)
    assert unsigned_url == f"{assets_config.base_url}/{path}"

    response = httpx.get(unsigned_url)
    assert response.status_code == 403

    # can access the file through the signed url
    signed_url = storage_client.get_url(path)
    assert signed_url != f"{assets_config.base_url}/{path}"
    assert signed_url.startswith(f"{assets_config.base_url}/{path}")

    response = httpx.get(signed_url)
    assert response.status_code == 200
    assert response.text == "hello world"

    # can delete the directory
    storage_client.delete_dataset_directory(dataset=DATASET)
    assert not storage_client.exists(DATASET)
    assert not storage_client.exists(path)
    response = httpx.get(signed_url)

    assert response.status_code == 200
    # ^ the file does not exist anymore, but it's still in the cache... we need to invalidate it
