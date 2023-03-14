# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import datetime
from contextlib import nullcontext as does_not_raise
from typing import Any, Dict, Optional

import jwt
import pytest

from api.jwt_token import is_jwt_valid, parse_jwt_public_key

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
    "keys,expectation",
    [
        (HUB_JWT_KEYS, does_not_raise()),
        ([], pytest.raises(Exception)),
        (UNSUPPORTED_ALGORITHM_JWT_KEYS, pytest.raises(Exception)),
    ],
)
def test_parse_jwk(
    keys: Any,
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
    assert (
        is_jwt_valid(
            dataset=DATASET_SEVERO_GLUE,
            token=HUB_JWT_TOKEN_FOR_SEVERO_GLUE,
            public_key=HUB_JWT_PUBLIC_KEY,
            algorithm=HUB_JWT_ALGORITHM,
            verify_exp=False,
            # This is a test token generated on 2023/03/14, so we don't want to verify the exp.
        )
        is True
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
public_key = """-----BEGIN PUBLIC KEY-----
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
algorithm_rs256 = "RS256"


@pytest.mark.parametrize(
    "public_key,payload,expected",
    [
        (None, payload_ok, False),
        (other_public_key, payload_ok, False),
        (public_key, {}, False),
        (public_key, {"sub": dataset_ok}, False),
        (public_key, {"sub": sub_wrong_1, "read": read_ok, "exp": exp_ok}, False),
        (public_key, {"sub": sub_wrong_2, "read": read_ok, "exp": exp_ok}, False),
        (public_key, {"sub": sub_wrong_3, "read": read_ok, "exp": exp_ok}, False),
        (public_key, {"sub": sub_wrong_4, "read": read_ok, "exp": exp_ok}, False),
        (public_key, {"sub": sub_ok, "read": read_wrong_1, "exp": exp_ok}, False),
        (public_key, {"sub": sub_ok, "read": read_wrong_2, "exp": exp_ok}, False),
        (public_key, {"sub": sub_ok, "read": read_ok, "exp": wrong_exp_1}, False),
        (public_key, {"sub": sub_ok, "read": read_ok, "exp": wrong_exp_2}, False),
        (public_key, payload_ok, True),
    ],
)
def test_is_jwt_valid(public_key: Optional[str], payload: Dict[str, str], expected: bool) -> None:
    token = jwt.encode(payload, private_key, algorithm=algorithm_rs256)
    assert is_jwt_valid(dataset=dataset_ok, token=token, public_key=public_key, algorithm=algorithm_rs256) is expected
