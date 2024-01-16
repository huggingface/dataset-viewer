# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import datetime
from dataclasses import dataclass, field

from botocore.signers import CloudFrontSigner
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


@dataclass
class CloudFront:
    """
    Signs CloudFront URLs using a private key.

    References:
    - https://github.com/boto/boto3/blob/develop/boto3/examples/cloudfront.rst
    - https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-creating-signed-url-canned-policy.html
    """

    _signer: CloudFrontSigner = field(init=False)

    def __init__(self, key_pair_id: str, private_key: str) -> None:
        """
        Args:
            key_pair_id (:obj:`str`): The cloudfront key pair id, eg. "K2JCJMDEHXQW5F"
            private_key (:obj:`str`): The cloudfront private key, in PEM format
        """
        pk = serialization.load_pem_private_key(private_key.encode("utf8"), password=None, backend=default_backend())
        p = padding.PKCS1v15()
        h = hashes.SHA1()

        def rsa_signer(message):
            return pk.sign(message, p, h)

        self._signer = CloudFrontSigner(key_pair_id, rsa_signer)

    def sign_url(self, url: str, date_less_than: datetime.datetime) -> str:
        """
        Create a signed url that will be valid until the specific expiry date
        provided using a canned policy.

        Args:
            url (:obj:`str`): The url to sign
            date_less_than (:obj:`datetime.datetime`): The expiry date

        Returns:
            :obj:`str`: The signed url
        """
        return self._signer.generate_presigned_url(url, date_less_than=date_less_than)
