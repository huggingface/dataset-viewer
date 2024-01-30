# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.


from datetime import datetime

import httpx
import pytest

from libcommon.cloudfront import get_cloudfront_signer
from libcommon.config import AssetsConfig, CloudFrontConfig, S3Config
from libcommon.storage_client import StorageClient

BUCKET = "hf-datasets-server-statics-test"
CLOUDFRONT_KEY_PAIR_ID = "K3814DK2QUJ71H"


def test_real_cloudfront(monkeypatch: pytest.MonkeyPatch) -> None:
    # this test is not mocked, and will fail if the credentials are not valid
    # it requires the following environment variables to be set:
    # - S3_ACCESS_KEY_ID
    # - S3_SECRET_ACCESS_KEY
    # - CLOUDFRONT_PRIVATE_KEY
    # To run it locally, set the environment variables in .env and run:
    #     set -a && source .env && set +a && make test
    monkeypatch.setenv("CLOUDFRONT_KEY_PAIR_ID", CLOUDFRONT_KEY_PAIR_ID)
    cloudfront_config = CloudFrontConfig.from_env()
    s3_config = S3Config.from_env()
    assets_config = AssetsConfig(
        base_url="https://datasets-server-test.us.dev.moon.huggingface.tech/assets",
        # ^ assets/ is hardcoded in cloudfront configuration
        storage_protocol="s3",
        storage_root=f"{BUCKET}/assets",
    )
    url_signer = get_cloudfront_signer(cloudfront_config=cloudfront_config)
    if not s3_config.access_key_id or not s3_config.secret_access_key or not url_signer:
        pytest.skip("the S3 and/or CloudFront credentials are not set in environment variables, so we skip the test")

    storage_client = StorageClient(
        protocol=assets_config.storage_protocol,
        storage_root=assets_config.storage_root,
        base_url=assets_config.base_url,
        overwrite=True,
        s3_config=s3_config,
        url_signer=url_signer,
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
    assert signed_url.startswith(f"{assets_config.base_url}/{path}?Expires=")
    assert "&Signature=" in signed_url
    assert signed_url.endswith(f"&Key-Pair-Id={CLOUDFRONT_KEY_PAIR_ID}")

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
