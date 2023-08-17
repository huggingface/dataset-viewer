# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import datetime
from contextlib import nullcontext as does_not_raise
from typing import Any, Dict, List, Optional

import jwt
import pytest

from libapi.config import ApiConfig
from libapi.exceptions import (
    JWTExpiredSignature,
    JWTInvalidClaimRead,
    JWTInvalidClaimSub,
    JWTInvalidKeyOrAlgorithm,
    JWTInvalidSignature,
    JWTMissingRequiredClaim,
)
from libapi.jwt_token import get_jwt_public_keys, parse_jwt_public_key, validate_jwt

HUB_JWT_KEYS = [{"crv": "Ed25519", "x": "-RBhgyNluwaIL5KFJb6ZOL2H1nmyI8mW4Z2EHGDGCXM", "kty": "OKP"}]
HUB_JWT_ALGORITHM = "EdDSA"
HUB_JWT_PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEA+RBhgyNluwaIL5KFJb6ZOL2H1nmyI8mW4Z2EHGDGCXM=
-----END PUBLIC KEY-----
"""
UNSUPPORTED_ALGORITHM_JWT_KEYS = [
    {
        "alg": "EC",
        "crv": "P-256",
        "x": "MKBCTNIcKUSDii11ySs3526iDZ8AiTo7Tu6KPAqv7D4",
        "y": "4Etl6SRW2YiLUrN5vfvVHuhp7x8PxltmWWlbbM4IFyM",
        "use": "enc",
        "kid": "1",
    }
]


@pytest.mark.parametrize(
    "keys_env_var,expected_keys",
    [
        ("", []),
        (
            (
                "-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEA+RBhgyNluwaIL5KFJb6ZOL2H1nmyI8mW4Z2EHGDGCXM=\n-----END"
                " PUBLIC KEY-----\n"
            ),
            [HUB_JWT_PUBLIC_KEY],
        ),
        (
            (
                "-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEA+RBhgyNluwaIL5KFJb6ZOL2H1nmyI8mW4Z2EHGDGCXM=\n-----END"
                " PUBLIC KEY-----\n,-----BEGIN PUBLIC"
                " KEY-----\nMCowBQYDK2VwAyEA+RBhgyNluwaIL5KFJb6ZOL2H1nmyI8mW4Z2EHGDGCXM=\n-----END PUBLIC KEY-----\n"
            ),
            [HUB_JWT_PUBLIC_KEY, HUB_JWT_PUBLIC_KEY],
        ),
    ],
)
def test_get_jwt_public_keys(keys_env_var: str, expected_keys: List[str]) -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("API_HF_JWT_ADDITIONAL_PUBLIC_KEYS", keys_env_var)
    api_config = ApiConfig.from_env(hf_endpoint="")
    assert get_jwt_public_keys(api_config) == expected_keys
    monkeypatch.undo()


@pytest.mark.parametrize(
    "keys,expectation",
    [
        (HUB_JWT_KEYS, does_not_raise()),
        ([], pytest.raises(Exception)),
        (UNSUPPORTED_ALGORITHM_JWT_KEYS, pytest.raises(Exception)),
    ],
)
def test_parse_jwk(
    keys: str,
    expectation: Any,
) -> None:
    with expectation:
        key = parse_jwt_public_key(keys=keys, hf_jwt_algorithm=HUB_JWT_ALGORITHM)
        assert key == HUB_JWT_PUBLIC_KEY


HUB_JWT_TOKEN_FOR_SEVERO_GLUE = (
    "eyJhbGciOiJFZERTQSJ9.eyJyZWFkIjp0cnVlLCJzdWIiOiJkYXRhc2V0cy9zZXZlcm8vZ2x1ZSIsImV4cCI6MTY3ODgwMjk0NH0"
    ".nIi1ZKinMBpYi4kKtirW-cQEt1cGnAziTGmJsZeN5UpE62jz4DcPaIPlSI5P5ciGOlTxy4SEhD1WITkQzpo3Aw"
)
DATASET_SEVERO_GLUE = "severo/glue"


def test_is_jwt_valid_with_ec() -> None:
    validate_jwt(
        dataset=DATASET_SEVERO_GLUE,
        token=HUB_JWT_TOKEN_FOR_SEVERO_GLUE,
        public_keys=[HUB_JWT_PUBLIC_KEY],
        algorithm=HUB_JWT_ALGORITHM,
        verify_exp=False,
        # This is a test token generated on 2023/03/14, so we don't want to verify the exp.
    )


private_key = """-----BEGIN RSA PRIVATE KEY-----
MIIBOQIBAAJAZTmplhS/Jd73ycVut7TglMObheQqXM7RZYlwazLU4wpfIVIwOh9I
sCZGSgLyFq42KWIikKLEs/yqx3pRGfq+rwIDAQABAkAMyF9WCICq86Eu5bO5lynV
H26AVfPTjHp87AI6R00C7p9n8hO/DhHaHpc3InOSsXsw9d2hmz37jwwBFiwMHMMh
AiEAtbttHlIO+yO29oXw4P6+yO11lMy1UpT1sPVTnR9TXbUCIQCOl7Zuyy2ZY9ZW
pDhW91x/14uXjnLXPypgY9bcfggJUwIhAJQG1LzrzjQWRUPMmgZKuhBkC3BmxhM8
LlwzmCXVjEw5AiA7JnAFEb9+q82T71d3q/DxD0bWvb6hz5ASoBfXK2jGBQIgbaQp
h4Tk6UJuj1xgKNs75Pk3pG2tj8AQiuBk3l62vRU=
-----END RSA PRIVATE KEY-----"""
public_key_ok = """-----BEGIN PUBLIC KEY-----
MFswDQYJKoZIhvcNAQEBBQADSgAwRwJAZTmplhS/Jd73ycVut7TglMObheQqXM7R
ZYlwazLU4wpfIVIwOh9IsCZGSgLyFq42KWIikKLEs/yqx3pRGfq+rwIDAQAB
-----END PUBLIC KEY-----"""
other_public_key = """-----BEGIN PUBLIC KEY-----
MFswDQYJKoZIhvcNAQEBBQADSgAwRwJAecoNIHMXczWkzTp9ePEcx6vPibrZVz/z
xYGX6G2jFcwFdsrO9nCecrtpSw5lwjW40aNVL9NL9yxPxDi2dyq4wQIDAQAB
-----END PUBLIC KEY-----"""


dataset_ok = "dataset"
wrong_dataset = "wrong_dataset"
exp_ok = datetime.datetime.now().timestamp() + 1000
wrong_exp_1 = datetime.datetime.now().timestamp() - 1000
wrong_exp_2 = 1
sub_ok = f"datasets/{dataset_ok}"
sub_wrong_1 = dataset_ok
sub_wrong_2 = f"dataset/{dataset_ok}"
sub_wrong_3 = f"models/{dataset_ok}"
sub_wrong_4 = f"datasets/{wrong_dataset}"
read_ok = True
read_wrong_1 = False
read_wrong_2 = "True"
payload_ok = {"sub": sub_ok, "read": read_ok, "exp": exp_ok}
algorithm_ok = "RS256"
algorithm_wrong = "HS256"


def encode_jwt(payload: Dict[str, Any]) -> str:
    return jwt.encode(payload, private_key, algorithm=algorithm_ok)


def assert_jwt(
    token: str, expectation: Any, public_keys: Optional[List[str]] = None, algorithm: str = algorithm_ok
) -> None:
    if public_keys is None:
        public_keys = [public_key_ok]
    with expectation:
        validate_jwt(dataset=dataset_ok, token=token, public_keys=public_keys, algorithm=algorithm)


@pytest.mark.parametrize(
    "public_keys,expectation",
    [
        ([other_public_key], pytest.raises(JWTInvalidSignature)),
        ([public_key_ok], does_not_raise()),
        ([public_key_ok, other_public_key], does_not_raise()),
        ([other_public_key, public_key_ok], does_not_raise()),
    ],
)
def test_validate_jwt_public_key(public_keys: List[str], expectation: Any) -> None:
    assert_jwt(encode_jwt(payload_ok), expectation, public_keys=public_keys)


@pytest.mark.parametrize(
    "algorithm,expectation",
    [
        (algorithm_wrong, pytest.raises(JWTInvalidKeyOrAlgorithm)),
        (algorithm_ok, does_not_raise()),
    ],
)
def test_validate_jwt_algorithm(algorithm: str, expectation: Any) -> None:
    assert_jwt(encode_jwt(payload_ok), expectation, algorithm=algorithm)


@pytest.mark.parametrize(
    "payload,expectation",
    [
        ({}, pytest.raises(JWTMissingRequiredClaim)),
        ({"sub": sub_ok}, pytest.raises(JWTMissingRequiredClaim)),
        ({"read": read_ok}, pytest.raises(JWTMissingRequiredClaim)),
        ({"exp": exp_ok}, pytest.raises(JWTMissingRequiredClaim)),
        ({"read": read_ok, "exp": exp_ok}, pytest.raises(JWTMissingRequiredClaim)),
        ({"sub": sub_ok, "exp": exp_ok}, pytest.raises(JWTMissingRequiredClaim)),
        ({"sub": sub_ok, "read": read_ok}, pytest.raises(JWTMissingRequiredClaim)),
        ({"sub": sub_ok, "read": read_ok, "exp": exp_ok}, does_not_raise()),
    ],
)
def test_validate_jwt_content_format(payload: Dict[str, str], expectation: Any) -> None:
    assert_jwt(encode_jwt(payload), expectation)


@pytest.mark.parametrize(
    "read,expectation",
    [
        (read_wrong_1, pytest.raises(JWTInvalidClaimRead)),
        (read_wrong_2, pytest.raises(JWTInvalidClaimRead)),
        (read_ok, does_not_raise()),
    ],
)
def test_validate_jwt_read(read: str, expectation: Any) -> None:
    assert_jwt(encode_jwt({"sub": sub_ok, "read": read, "exp": exp_ok}), expectation)


@pytest.mark.parametrize(
    "sub,expectation",
    [
        (sub_wrong_1, pytest.raises(JWTInvalidClaimSub)),
        (sub_wrong_2, pytest.raises(JWTInvalidClaimSub)),
        (sub_wrong_3, pytest.raises(JWTInvalidClaimSub)),
        (sub_wrong_4, pytest.raises(JWTInvalidClaimSub)),
        (sub_ok, does_not_raise()),
    ],
)
def test_validate_jwt_subject(sub: str, expectation: Any) -> None:
    assert_jwt(encode_jwt({"sub": sub, "read": read_ok, "exp": exp_ok}), expectation)


@pytest.mark.parametrize(
    "expiration,expectation",
    [
        (wrong_exp_1, pytest.raises(JWTExpiredSignature)),
        (wrong_exp_2, pytest.raises(JWTExpiredSignature)),
        (exp_ok, does_not_raise()),
    ],
)
def test_validate_jwt_expiration(expiration: str, expectation: Any) -> None:
    assert_jwt(
        encode_jwt({"sub": sub_ok, "read": read_ok, "exp": expiration}),
        expectation,
    )
