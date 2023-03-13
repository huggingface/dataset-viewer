# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import datetime
from contextlib import nullcontext as does_not_raise
from typing import Any, Dict, Optional

import jwt
import pytest

from api.jwt_token import is_jwt_valid, parse_jwt_public_key

HUB_JWT_KEYS = [{"crv": "Ed25519", "x": "-RBhgyNluwaIL5KFJb6ZOL2H1nmyI8mW4Z2EHGDGCXM", "kty": "OKP"}]
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
        parse_jwt_public_key(keys=keys, hf_jwt_algorithm="EdDSA")


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
exp = datetime.datetime.now().timestamp() + 1000
wrong_exp_1 = datetime.datetime.now().timestamp() - 1000
wrong_exp_2 = 1
sub_ok = {"repoName": dataset_ok, "repoType": "dataset", "read": True}
sub_wrong_format_1 = dataset_ok
sub_wrong_format_2 = {"repoName": dataset_ok}
sub_wrong_value_1 = {"repoName": wrong_dataset, "repoType": "dataset", "read": True}
sub_wrong_value_2 = {"repoName": dataset_ok, "repoType": "model", "read": True}
sub_wrong_value_3 = {"repoName": dataset_ok, "repoType": "dataset", "read": False}
payload_ok = {"sub": sub_ok, "exp": exp}
algorithm_rs256 = "RS256"


@pytest.mark.parametrize(
    "public_key,payload,expected",
    [
        (None, payload_ok, False),
        (other_public_key, payload_ok, False),
        (public_key, {}, False),
        (public_key, {"sub": dataset_ok}, False),
        (public_key, {"exp": exp}, False),
        (public_key, {"sub": wrong_dataset, "exp": exp}, False),
        (public_key, {"sub": dataset_ok, "exp": wrong_exp_1}, False),
        (public_key, {"sub": dataset_ok, "exp": wrong_exp_2}, False),
        (public_key, {"sub": sub_wrong_format_1, "exp": exp}, False),
        (public_key, {"sub": sub_wrong_format_2, "exp": exp}, False),
        (public_key, {"sub": sub_wrong_value_1, "exp": exp}, False),
        (public_key, {"sub": sub_wrong_value_2, "exp": exp}, False),
        (public_key, {"sub": sub_wrong_value_3, "exp": exp}, False),
        (public_key, payload_ok, True),
    ],
)
def test_is_jwt_valid(public_key: Optional[str], payload: Dict[str, str], expected: bool) -> None:
    token = jwt.encode(payload, private_key, algorithm=algorithm_rs256)
    assert is_jwt_valid(dataset=dataset_ok, token=token, public_key=public_key, algorithm=algorithm_rs256) is expected
