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

from libapi.exceptions import (
    JWKError,
    JWTExpiredSignature,
    JWTInvalidClaimRead,
    JWTInvalidClaimSub,
    JWTInvalidKeyOrAlgorithm,
    JWTInvalidSignature,
    JWTMissingRequiredClaim,
    UnexpectedApiError,
)

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


def validate_jwt(dataset: str, token: Any, public_key: str, algorithm: str, verify_exp: Optional[bool] = True) -> None:
    """
    Check if the JWT is valid for the dataset.

    The JWT is decoded with the public key, and the payload must be:
      {"sub": "datasets/<...dataset identifier...>", "read": true, "exp": <...date...>}

    Raise an exception if any of the condition is not met.

    Args:
        dataset (str): the dataset identifier
        token (Any): the JWT token to decode
        public_key (str): the public key to use to decode the JWT token
        algorithm (str): the algorithm to use to decode the JWT token
        verify_exp (bool|None): whether to verify the expiration of the JWT token. Default to True.

    Raise:

    """
    try:
        decoded = jwt.decode(
            jwt=token,
            key=public_key,
            algorithms=[algorithm],
            options={"require": ["exp", "sub", "read"], "verify_exp": verify_exp},
        )
        logging.debug(f"Decoded JWT is: '{public_key}'.")
    except jwt.exceptions.MissingRequiredClaimError as e:
        raise JWTMissingRequiredClaim("A claim is missing in the JWT payload.", e) from e
    except jwt.exceptions.ExpiredSignatureError as e:
        raise JWTExpiredSignature("The JWT signature has expired. Try to refresh the token.", e) from e
    except jwt.exceptions.InvalidSignatureError as e:
        raise JWTInvalidSignature(
            "The JWT signature verification failed. Check the signing key and the algorithm.", e
        ) from e
    except (jwt.exceptions.InvalidKeyError, jwt.exceptions.InvalidAlgorithmError) as e:
        raise JWTInvalidKeyOrAlgorithm(
            (
                "The key used to verify the signature is not compatible with the algorithm. Check the signing key and"
                " the algorithm."
            ),
            e,
        ) from e
    except Exception as e:
        logging.debug(
            f"Missing public key '{public_key}' or algorithm '{algorithm}' to decode JWT token. Skipping JWT"
            " validation."
        )
        raise UnexpectedApiError("An error has occurred while decoding the JWT.", e) from e
    sub = decoded.get("sub")
    if not isinstance(sub, str) or not sub.startswith("datasets/") or sub.removeprefix("datasets/") != dataset:
        raise JWTInvalidClaimSub(
            "The 'sub' claim in JWT payload is invalid. It should be in the form 'datasets/<...dataset"
            " identifier...>'."
        )
    read = decoded.get("read")
    if read is not True:
        raise JWTInvalidClaimRead("The 'read' claim in JWT payload is invalid. It should be set to 'true'.")
