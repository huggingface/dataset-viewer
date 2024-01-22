# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal, Optional, Union

from botocore.signers import CloudFrontSigner
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.hazmat.primitives.hashes import SHA1
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from datasets import Audio, Features, Image
from datasets.features.features import FeatureType, Sequence

from libcommon.config import CloudFrontConfig
from libcommon.utils import FeatureItem, get_expires


class InvalidFirstRowsError(ValueError):
    pass


VisitPath = list[Union[str, Literal[0]]]


@dataclass
class AssetUrlPath:
    feature_type: Literal["Audio", "Image"]
    path: VisitPath


# invert to_features_list
def to_features_dict(features: list[FeatureItem]) -> Features:
    return Features({feature_item["name"]: feature_item["type"] for feature_item in features})


def _visit(
    feature: FeatureType, func: Callable[[FeatureType, VisitPath], Optional[FeatureType]], visit_path: VisitPath = []
) -> FeatureType:
    """Visit a (possibly nested) feature.

    Args:
        feature (FeatureType): the feature type to be checked
    Returns:
        visited feature (FeatureType)
    """
    if isinstance(feature, dict):
        out = func({k: _visit(f, func, visit_path + [k]) for k, f in feature.items()}, visit_path)
    elif isinstance(feature, (list, tuple)):
        out = func([_visit(feature[0], func, visit_path + [0])], visit_path)
    elif isinstance(feature, Sequence):
        out = func(Sequence(_visit(feature.feature, func, visit_path + [0]), length=feature.length), visit_path)
    else:
        out = func(feature, visit_path)
    return feature if out is None else out


def get_asset_url_paths(features: Features) -> list[AssetUrlPath]:
    asset_url_paths: list[AssetUrlPath] = []
    for column, feature in features.items():

        def classify(feature: FeatureType, visit_path: VisitPath) -> None:
            if isinstance(feature, Image):
                asset_url_paths.append(AssetUrlPath(feature_type="Image", path=visit_path))
            elif isinstance(feature, Audio):
                asset_url_paths.append(AssetUrlPath(feature_type="Audio", path=visit_path + [0]))

        _visit(feature, classify, [column])
    return asset_url_paths


class URLSigner(ABC):
    @abstractmethod
    def sign_url(self, url: str) -> str:
        pass

    def _sign_asset_url_path(self, cell: Any, asset_url_path: AssetUrlPath) -> Any:
        if len(asset_url_path.path) == 0:
            if not isinstance(cell, dict):
                raise InvalidFirstRowsError("Expected the cell to be a dict")
            src = cell.get("src")
            if not isinstance(src, str):
                raise InvalidFirstRowsError('Expected cell["src"] to be a string')
            cell["src"] = self.sign_url(url=src)
            # ^ sign the url in place
        else:
            key = asset_url_path.path.pop(0)
            if key == 0:
                # it's a list, we have to sign each element
                if not isinstance(cell, list):
                    raise InvalidFirstRowsError("Expected the cell to be a list")
                for cell_item in cell:
                    self._sign_asset_url_path(cell=cell_item, asset_url_path=asset_url_path)
            else:
                # it's a dict, we have to sign the value of the key
                if not isinstance(cell, dict):
                    raise InvalidFirstRowsError("Expected the cell to be a dict")
                cell[key] = self._sign_asset_url_path(cell=cell[key], asset_url_path=asset_url_path)

    def sign_urls_in_first_rows(self, first_rows: Any) -> Any:
        if not isinstance(first_rows, dict):
            raise InvalidFirstRowsError("Expected response to be a dict")
        # parse the features to find the paths to assets URLs
        features_list = first_rows.get("features")
        if not isinstance(features_list, list):
            raise InvalidFirstRowsError('Expected response["features"] a list')
        features_dict = to_features_dict(features_list)
        features = Features.from_dict(features_dict)
        asset_url_paths = get_asset_url_paths(features)
        if not asset_url_paths:
            return first_rows
        # sign the URLs
        row_items = first_rows.get("rows")
        if not isinstance(row_items, list):
            raise InvalidFirstRowsError('Expected response["rows"] to be a list')
        for row_item in row_items:
            if not isinstance(row_item, dict):
                raise InvalidFirstRowsError('Expected response["rows"][i] to be a dict')
            row = row_item.get("row")
            if not isinstance(row, dict):
                raise InvalidFirstRowsError('Expected response["rows"][i]["row"] to be a dict')
            for asset_url_path in asset_url_paths:
                self._sign_asset_url_path(cell=row, asset_url_path=asset_url_path)
        return first_rows

    def sign_urls_in_obj(self, obj: Any, base_url: str) -> Any:
        if isinstance(obj, dict):
            return {k: self.sign_urls_in_obj(v, base_url) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.sign_urls_in_obj(v, base_url) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.sign_urls_in_obj(v, base_url) for v in obj)
        elif isinstance(obj, str) and obj.startswith(base_url + "/"):
            return self.sign_url(url=obj)
        else:
            return obj


class InvalidPrivateKeyError(ValueError):
    pass


padding = PKCS1v15()
algorithm = SHA1()  # nosec
# ^ bandit raises a warning:
#     https://bandit.readthedocs.io/en/1.7.5/blacklists/blacklist_calls.html#b303-md5
#   but CloudFront mandates SHA1


class CloudFront(URLSigner):
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


def get_url_signer(cloudfront_config: CloudFrontConfig) -> Optional[URLSigner]:
    return (
        CloudFront(
            key_pair_id=cloudfront_config.key_pair_id,
            private_key=cloudfront_config.private_key,
            expiration_seconds=cloudfront_config.expiration_seconds,
        )
        if cloudfront_config.key_pair_id and cloudfront_config.private_key
        else None
    )
