# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Any, Optional, Union

import jwt
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ec import (
    EllipticCurvePrivateKey,
    EllipticCurvePublicKey,
)
from cryptography.hazmat.primitives.asymmetric.ed448 import (
    Ed448PrivateKey,
    Ed448PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey
from jwt.algorithms import (
    ECAlgorithm,
    HMACAlgorithm,
    OKPAlgorithm,
    RSAAlgorithm,
    RSAPSSAlgorithm,
)

from libapi.exceptions import JWKError

ASYMMETRIC_ALGORITHMS = (ECAlgorithm, OKPAlgorithm, RSAAlgorithm, RSAPSSAlgorithm)
SYMMETRIC_ALGORITHMS = (HMACAlgorithm,)


def is_public_key(
    key: Union[
        EllipticCurvePublicKey,
        EllipticCurvePrivateKey,
        Ed25519PublicKey,
        Ed448PublicKey,
        Ed25519PrivateKey,
        Ed448PrivateKey,
        RSAPrivateKey,
        RSAPublicKey,
    ]
) -> bool:
    return hasattr(key, "public_bytes")


def parse_jwt_public_key(keys: Any, hf_jwt_algorithm: str) -> str:
    """parse the input JSON to extract the public key

    Note that the return type is Any in order not to enter in too much details. See
    https://github.com/jpadilla/pyjwt/blob/777efa2f51249f63b0f95804230117723eca5d09/jwt/algorithms.py#L629-L651
    In our case, the type should be cryptography.hazmat.backends.openssl.ed25519._Ed25519PublicKey

    Args:
        keys (Any): the JSON to parse
        hf_jwt_algorithm (str): the JWT algorithm to use.

    Returns:
        str: the public key
    """
    try:
        expected_algorithm = jwt.get_algorithm_by_name(hf_jwt_algorithm)
        if not isinstance(expected_algorithm, (*ASYMMETRIC_ALGORITHMS, *SYMMETRIC_ALGORITHMS)):
            raise NotImplementedError()
    except NotImplementedError as err:
        raise RuntimeError(f"Invalid algorithm for JWT verification: {hf_jwt_algorithm} is not supported") from err

    if not isinstance(keys, list):
        raise ValueError("Payload from moon must be a list of JWK formatted keys.")
    try:
        key = expected_algorithm.from_jwk(keys[0])
        if not isinstance(expected_algorithm, ASYMMETRIC_ALGORITHMS) or isinstance(key, bytes):
            return key.decode("utf-8")
        if not is_public_key(key):
            raise RuntimeError("Failed to parse JWT key: the provided key is a private key")
        return key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")
        # ^ we assume that the key contain UTF-8 encoded bytes, which is why we use type ignore for mypy
    except (jwt.InvalidKeyError, KeyError) as err:
        raise RuntimeError(f"Failed to parse JWT key: {err.args[0]}") from err


def fetch_jwt_public_key(
    url: str,
    hf_jwt_algorithm: str,
    hf_timeout_seconds: Optional[float] = None,
) -> str:
    """fetch the public key to decode the JWT token from the input URL

    See https://huggingface.co/api/keys/jwt

    Args:
        url (str): the URL to fetch the public key from
        hf_jwt_algorithm (str): the JWT algorithm to use.
        hf_timeout_seconds (float|None): the timeout in seconds for the external authentication service. It
            is used both for the connection timeout and the read timeout. If None, the request never timeouts.

    Returns:
        str: the public key
    """
    try:
        response = requests.get(url, timeout=hf_timeout_seconds)
        response.raise_for_status()
        return parse_jwt_public_key(keys=response.json(), hf_jwt_algorithm=hf_jwt_algorithm)
    except Exception as err:
        raise JWKError(f"Failed to fetch or parse the JWT public key from {url}. ", cause=err) from err


def is_jwt_valid(
    dataset: str, token: Any, public_key: Optional[str], algorithm: Optional[str], verify_exp: Optional[bool] = True
) -> bool:
    """
    Check if the JWT is valid for the dataset.

    The JWT is decoded with the public key, and the "sub" claim must be:
      {"repoName": <...>, "repoType": "dataset", "read": true}
    where <...> is the dataset identifier.

    Returns True only if all the conditions are met. Else, it returns False.

    Args:
        dataset (str): the dataset identifier
        token (Any): the JWT token to decode
        public_key (str|None): the public key to use to decode the JWT token
        algorithm (str|None): the algorithm to use to decode the JWT token
        verify_exp (bool|None): whether to verify the expiration of the JWT token. Default to True.

    Returns:
        bool: True if the JWT is valid for the input dataset, else False
    """
    if not public_key or not algorithm:
        logging.debug(
            f"Missing public key '{public_key}' or algorithm '{algorithm}' to decode JWT token. Skipping JWT"
            " validation."
        )
        return False
    try:
        decoded = jwt.decode(
            jwt=token,
            key=public_key,
            algorithms=[algorithm],
            options={"require": ["exp", "sub", "read"], "verify_exp": verify_exp},
        )
        logging.debug(f"Decoded JWT is: '{public_key}'.")
    except Exception:
        logging.debug(
            f"Missing public key '{public_key}' or algorithm '{algorithm}' to decode JWT token. Skipping JWT"
            " validation."
        )
        return False
    sub = decoded.get("sub")
    if not isinstance(sub, str) or not sub.startswith("datasets/") or sub.removeprefix("datasets/") != dataset:
        return False
    read = decoded.get("read")
    return read is True
