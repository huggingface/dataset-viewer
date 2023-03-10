# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Any, Optional

import jwt
import requests

from api.utils import JWKError


def parse_jwk(
    json: Any,
    hf_jwt_algorithm: str,
) -> Any:
    """parse the input JSON to extract the public key

    Note that the return type is Any in order not to enter in too much details. See
    https://github.com/jpadilla/pyjwt/blob/777efa2f51249f63b0f95804230117723eca5d09/jwt/algorithms.py#L629-L651
    In our case, the type should be cryptography.hazmat.backends.openssl.ed25519._Ed25519PublicKey

    Args:
        json (Any): the JSON to parse
        hf_jwt_algorithm (str): the JWT algorithm to use.

    Returns:
        Any: the public key
    """
    try:
        expected_algorithm = jwt.get_algorithm_by_name(hf_jwt_algorithm)
    except NotImplementedError as err:
        raise RuntimeError(f"Invalid algorithm for JWT verification: {hf_jwt_algorithm}") from err

    try:
        return expected_algorithm.from_jwk(json[0])
    except (jwt.InvalidKeyError, KeyError) as err:
        raise RuntimeError("Failed to parse JWT key") from err


def fetch_jwt_public_key(
    url: str,
    hf_jwt_algorithm: str,
    hf_timeout_seconds: Optional[float] = None,
) -> Any:
    """fetch the public key to decode the JWT token from the input URL

    See https://huggingface.co/api/keys/jwt

    Args:
        url (str): the URL to fetch the public key from
        hf_jwt_algorithm (str): the JWT algorithm to use.
        hf_timeout_seconds (float|None): the timeout in seconds for the external authentication service. It
            is used both for the connection timeout and the read timeout. If None, the request never timeouts.

    Returns:
        Any: the public key
    """
    try:
        response = requests.get(url, timeout=hf_timeout_seconds)
        response.raise_for_status()
        return parse_jwk(json=response.json(), hf_jwt_algorithm=hf_jwt_algorithm)
    except Exception as err:
        raise JWKError(f"Failed to fetch or parse the JWT public key from {url}. ", cause=err) from err


def is_jwt_valid(dataset: str, token: Any, public_key: Optional[Any], algorithm: Optional[str]) -> bool:
    """
    Check if the JWT is valid for the input dataset.

    The JWT is decoded with the public key, and the "sub" claim must be equal to the input dataset identifier.

    Returns True only if all the conditions are met. Else, it returns False.

    Args:
        dataset (str): the dataset identifier
        token (Any): the JWT token to decode
        public_key (Any|None): the public key to use to decode the JWT token
        algorithm (str|None): the algorithm to use to decode the JWT token

    Returns:
        bool: True if the JWT is valid for the input dataset, else False
    """
    if not public_key or not algorithm:
        return False
    try:
        decoded = jwt.decode(jwt=token, key=public_key, algorithms=[algorithm], options={"require": ["exp", "sub"]})
    except Exception as err:
        print(err)
        return False
    sub = decoded.get("sub")
    return isinstance(sub, str) and sub == dataset
