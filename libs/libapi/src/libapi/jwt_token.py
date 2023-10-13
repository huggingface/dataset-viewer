# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Any, Optional, Union

import httpx
import jwt
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
    JWTExpiredSignature,
    JWTInvalidClaimRead,
    JWTInvalidClaimSub,
    JWTInvalidKeyOrAlgorithm,
    JWTInvalidSignature,
    JWTKeysError,
    JWTMissingRequiredClaim,
    UnexpectedApiError,
)

ASYMMETRIC_ALGORITHMS = (ECAlgorithm, OKPAlgorithm, RSAAlgorithm, RSAPSSAlgorithm)
SYMMETRIC_ALGORITHMS = (HMACAlgorithm,)
SupportedAlgorithm = Union[ECAlgorithm, OKPAlgorithm, RSAAlgorithm, RSAPSSAlgorithm, HMACAlgorithm]
SupportedKey = Union[
    Ed448PrivateKey,
    Ed448PublicKey,
    Ed25519PrivateKey,
    Ed25519PublicKey,
    EllipticCurvePrivateKey,
    EllipticCurvePublicKey,
    RSAPrivateKey,
    RSAPublicKey,
    bytes,
]


def is_public_key(key: SupportedKey) -> bool:
    return hasattr(key, "public_bytes")


def create_algorithm(algorithm_name: str) -> SupportedAlgorithm:
    """
    Create an algorithm object from the algorithm name.

    Args:
        algorithm_name (str): the algorithm name

    Returns:
        SupportedAlgorithm: the algorithm object

    Raises:
        RuntimeError: if the algorithm is not supported
    """
    try:
        algorithm = jwt.get_algorithm_by_name(algorithm_name)
        if not isinstance(algorithm, (*ASYMMETRIC_ALGORITHMS, *SYMMETRIC_ALGORITHMS)):
            raise NotImplementedError()
    except NotImplementedError as err:
        raise RuntimeError(f"Invalid algorithm for JWT verification: {algorithm_name} is not supported") from err
    return algorithm


def _key_to_pem(key: SupportedKey, algorithm: SupportedAlgorithm) -> str:
    """
    Convert the key to PEM format.

    Args:
        key (SupportedKey): the key to convert

    Returns:
        str: the key in PEM format (PKCS#8)

    Raises:
        RuntimeError: if the key is not a public key
    """
    if isinstance(algorithm, SYMMETRIC_ALGORITHMS) or isinstance(key, bytes):
        return key.decode("utf-8")  # type: ignore
    if not is_public_key(key):
        raise RuntimeError("Failed to parse JWT key: the provided key is a private key")
    return key.public_bytes(  # type: ignore
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    # ^ we assume that the key contain UTF-8 encoded bytes, which is why we use type ignore for mypy


def parse_jwt_public_key_json(payload: Any, algorithm: SupportedAlgorithm) -> str:
    """
    Parse the payload (JSON format) to extract the public key, validating that it's a public key, and that it is
    compatible with the algorithm

    Args:
        keys (Any): the JSON to parse. It must be a list of keys in JWK format
        algorithm (SupportedAlgorithm): the algorithm the key should implement

    Returns:
        str: the public key in PEM format

    Raises:
        RuntimeError: if the payload is not compatible with the algorithm, or if the key is not public
        ValueError: if the input is not a list
    """
    if not isinstance(payload, list) or not payload:
        raise ValueError("Payload must be a list of JWK formatted keys.")
    try:
        key = algorithm.from_jwk(payload[0])
    except (jwt.InvalidKeyError, KeyError) as err:
        raise RuntimeError(f"Failed to parse JWT key: {err.args[0]}") from err
    return _key_to_pem(key, algorithm)


def parse_jwt_public_key_pem(payload: str, algorithm: SupportedAlgorithm) -> str:
    """
    Parse the input string to validate it's a public key in PEM format, and that it is compatible
    with the algorithm

    Args:
        key (str): the key to parse. It should be a public key in PEM format
        algorithm (SupportedAlgorithm): the algorithm the key should implement

    Returns:
        str: the public key in PEM format

    Raises:
        RuntimeError: if the payload is not compatible with the algorithm, or if the key is not public
        ValueError: if the input is not a list
    """
    try:
        key = algorithm.prepare_key(payload)
    except (jwt.InvalidKeyError, KeyError) as err:
        raise RuntimeError(f"Failed to parse JWT key: {err.args[0]}") from err
    return _key_to_pem(key, algorithm)


def fetch_jwt_public_key_json(
    url: str,
    hf_timeout_seconds: Optional[float] = None,
) -> Any:
    """
    Fetch the public key from the input URL

    See https://huggingface.co/api/keys/jwt

    Args:
        url (str): the URL to fetch the public key from
        hf_timeout_seconds (float|None): the timeout in seconds for the external authentication service. It
            is used both for the connection timeout and the read timeout. If None, the request never timeouts.

    Returns:
        Any: the response JSON payload

    Raises:
        RuntimeError: if the request fails
    """
    try:
        response = httpx.get(url, timeout=hf_timeout_seconds)
        response.raise_for_status()
        return response.json()
    except Exception as err:
        raise RuntimeError(f"Failed to fetch the JWT public key from {url}. ") from err


def get_jwt_public_keys(
    algorithm_name: Optional[str] = None,
    public_key_url: Optional[str] = None,
    additional_public_keys: Optional[list[str]] = None,
    timeout_seconds: Optional[float] = None,
) -> list[str]:
    """
    Get the public keys to use to decode the JWT token.

    The keys can be created by two mechanisms:
    - one key can be fetched from the public_key_url (must be in JWK format, ie. JSON)
    - additional keys can be provided as a list of PEM formatted keys
    All of these keys are then converted to PEM format (PKCS#8) and returned as a list. The remote key is first.
    The keys must be compatible with the algorithm.

    Args:
        algorithm_name (str|None): the algorithm to use to decode the JWT token. If not provided, no keys will be
          returned
        public_key_url (str|None): the URL to fetch the public key from
        additional_public_keys (list[str]|None): additional public keys to use to decode the JWT token
        timeout_seconds (float|None): the timeout in seconds for fetching the remote key

    Returns:
        list[str]: the list of public keys in PEM format

    Raises:
        JWTKeysError: if some exception occurred while creating the public keys. Some reasons: if the algorithm
          is not supported, if a payload could not be parsed, if a key is not compatible with the algorithm,
            if a key is not public, if th remote key could not be fetch or parsed
    """
    try:
        keys: list[str] = []
        if not algorithm_name:
            return keys
        algorithm = create_algorithm(algorithm_name)
        if public_key_url:
            payload = fetch_jwt_public_key_json(
                url=public_key_url,
                hf_timeout_seconds=timeout_seconds,
            )
            keys.append(parse_jwt_public_key_json(payload=payload, algorithm=algorithm))
        if additional_public_keys:
            keys.extend(
                parse_jwt_public_key_pem(payload=payload, algorithm=algorithm) for payload in additional_public_keys
            )
        logging.debug(f"JWT public keys are: {', '.join(keys)}.")
        return keys
    except Exception as err:
        raise JWTKeysError("Failed to create the JWT public keys.") from err


def validate_jwt(
    dataset: str, token: Any, public_keys: list[str], algorithm: str, verify_exp: Optional[bool] = True
) -> None:
    """
    Check if the JWT is valid for the dataset.

    The JWT is decoded with the public key, and the payload must be:
      {"sub": "datasets/<...dataset identifier...>", "read": true, "exp": <...date...>}
    or
      {"sub": "/datasets/<...dataset identifier...>", "read": true, "exp": <...date...>}

    Raise an exception if any of the condition is not met.

    Args:
        dataset (str): the dataset identifier
        token (Any): the JWT token to decode
        public_keys (list[str]): the public keys to use to decode the JWT token. They are tried in order.
        algorithm (str): the algorithm to use to decode the JWT token
        verify_exp (bool|None): whether to verify the expiration of the JWT token. Default to True.

    Raise:
        JWTInvalidSignature: if the signature verification failed
        JWTMissingRequiredClaim: if a claim is missing in the JWT payload
        JWTExpiredSignature: if the JWT signature has expired
        JWTInvalidKeyOrAlgorithm: if the key used to verify the signature is not compatible with the algorithm
        JWTInvalidClaimSub: if the 'sub' claim in JWT payload is invalid
        JWTInvalidClaimRead: if the 'read' claim in JWT payload is invalid
        UnexpectedApiError: if another error occurred while decoding the JWT
    """
    for public_key in public_keys:
        logging.debug(f"Trying to decode the JWT with key #{public_keys.index(public_key)}: {public_key}.")
        try:
            decoded = jwt.decode(
                jwt=token,
                key=public_key,
                algorithms=[algorithm],
                options={"require": ["exp", "sub", "read"], "verify_exp": verify_exp},
            )
            logging.debug(f"Decoded JWT is: '{public_key}'.")
            break
        except jwt.exceptions.InvalidSignatureError as e:
            if public_key == public_keys[-1]:
                raise JWTInvalidSignature(
                    "The JWT signature verification failed. Check the signing key and the algorithm.", e
                ) from e
            logging.debug(f"JWT signature verification failed with key: '{public_key}'. Trying next key.")
        except jwt.exceptions.MissingRequiredClaimError as e:
            raise JWTMissingRequiredClaim("A claim is missing in the JWT payload.", e) from e
        except jwt.exceptions.ExpiredSignatureError as e:
            raise JWTExpiredSignature("The JWT signature has expired. Try to refresh the token.", e) from e

        except (jwt.exceptions.InvalidKeyError, jwt.exceptions.InvalidAlgorithmError) as e:
            raise JWTInvalidKeyOrAlgorithm(
                (
                    "The key used to verify the signature is not compatible with the algorithm. Check the signing key"
                    " and the algorithm."
                ),
                e,
            ) from e
        except Exception as e:
            raise UnexpectedApiError("An error has occurred while decoding the JWT.", e) from e
    sub = decoded.get("sub")
    if not isinstance(sub, str) or (
        (not sub.startswith("datasets/") or sub.removeprefix("datasets/") != dataset)
        and (not sub.startswith("/datasets/") or sub.removeprefix("/datasets/") != dataset)
    ):
        raise JWTInvalidClaimSub(
            "The 'sub' claim in JWT payload is invalid. It should be in the form 'datasets/<...dataset"
            " identifier...>' or '/datasets/<...dataset identifier...>'."
        )
    read = decoded.get("read")
    if read is not True:
        raise JWTInvalidClaimRead("The 'read' claim in JWT payload is invalid. It should be set to 'true'.")
