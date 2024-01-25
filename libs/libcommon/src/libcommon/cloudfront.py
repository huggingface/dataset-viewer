# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import datetime
from functools import partial

from botocore.signers import CloudFrontSigner
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.hazmat.primitives.hashes import SHA1
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from libcommon.utils import get_expires


class InvalidPrivateKeyError(ValueError):
    pass


padding = PKCS1v15()
algorithm = SHA1()  # nosec
# ^ bandit raises a warning:
#     https://bandit.readthedocs.io/en/1.7.5/blacklists/blacklist_calls.html#b303-md5
#   but CloudFront mandates SHA1


class CloudFront:
    """
    Signs CloudFront URLs using a private key.

    References:
    - https://github.com/boto/boto3/blob/develop/boto3/examples/cloudfront.rst
    - https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-creating-signed-url-canned-policy.html
    """

    _expiration_seconds: int
    _signer: CloudFrontSigner

    def __init__(self, key_pair_id: str, private_key: str, expiration_seconds: int) -> None:
        """
        Args:
            key_pair_id (`str`): The cloudfront key pair id, eg. "K2JCJMDEHXQW5F"
            private_key (`str`): The cloudfront private key, in PEM format
            expiration_seconds (`int`): The number of seconds the signed url will be valid for
        """
        try:
            pk = load_pem_private_key(private_key.encode("utf8"), password=None, backend=default_backend())
        except ValueError as e:
            raise InvalidPrivateKeyError("Invalid private key") from e
        if not isinstance(pk, RSAPrivateKey):
            raise InvalidPrivateKeyError("Expected an RSA private key")

        self._expiration_seconds = expiration_seconds
        self._signer = CloudFrontSigner(key_pair_id, partial(pk.sign, padding=padding, algorithm=algorithm))

    def _sign_url(self, url: str, date_less_than: datetime.datetime) -> str:
        """
        Create a signed url that will be valid until the specific expiry date
        provided using a canned policy.

        Args:
            url (`str`): The url to sign
            date_less_than (`datetime.datetime`): The expiry date

        Returns:
            `str`: The signed url
        """
        return self._signer.generate_presigned_url(url, date_less_than=date_less_than)  # type: ignore
        # ^ ignoring mypy type error, it should return a string

    def sign_url(self, url: str) -> str:
        """
        Create a signed url that will be valid until the configured delay has expired
        using a canned policy.

        Args:
            url (`str`): The url to sign

        Returns:
            `str`: The signed url
        """
        date_less_than = get_expires(seconds=self._expiration_seconds)
        return self._sign_url(url=url, date_less_than=date_less_than)
