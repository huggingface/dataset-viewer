# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import datetime
from contextlib import nullcontext as does_not_raise
from typing import Any, Optional
from unittest.mock import patch

import jwt
import pytest
from ecdsa import Ed25519, SigningKey

from libapi.config import ApiConfig
from libapi.exceptions import (
    JWTExpiredSignature,
    JWTInvalidClaimRead,
    JWTInvalidClaimSub,
    JWTInvalidKeyOrAlgorithm,
    JWTInvalidSignature,
    JWTKeysError,
    JWTMissingRequiredClaim,
)
from libapi.jwt_token import (
    create_algorithm,
    get_jwt_public_keys,
    parse_jwt_public_key_json,
    parse_jwt_public_key_pem,
    validate_jwt,
)

algorithm_name_eddsa = "EdDSA"
algorithm_name_rs256 = "RS256"
algorithm_name_hs256 = "HS256"
algorithm_name_unknown = "unknown"


@pytest.mark.parametrize(
    "algorithm_name,expectation",
    [
        (algorithm_name_eddsa, does_not_raise()),
        (algorithm_name_rs256, does_not_raise()),
        (algorithm_name_hs256, does_not_raise()),
        (algorithm_name_unknown, pytest.raises(RuntimeError)),
    ],
)
def test_create_algorithm(algorithm_name: str, expectation: Any) -> None:
    with expectation:
        create_algorithm(algorithm_name)


algorithm_eddsa = create_algorithm(algorithm_name_eddsa)
eddsa_public_key_json_payload = {"crv": "Ed25519", "x": "-RBhgyNluwaIL5KFJb6ZOL2H1nmyI8mW4Z2EHGDGCXM", "kty": "OKP"}
# ^ given by https://huggingface.co/api/keys/jwt (as of 2023/08/18)
eddsa_public_key_pem = """-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEA+RBhgyNluwaIL5KFJb6ZOL2H1nmyI8mW4Z2EHGDGCXM=
-----END PUBLIC KEY-----
"""
another_algorithm_public_key_json_payload = {
    "alg": "EC",
    "crv": "P-256",
    "x": "MKBCTNIcKUSDii11ySs3526iDZ8AiTo7Tu6KPAqv7D4",
    "y": "4Etl6SRW2YiLUrN5vfvVHuhp7x8PxltmWWlbbM4IFyM",
    "use": "enc",
    "kid": "1",
}


@pytest.mark.parametrize(
    "payload,expected_pem,expectation",
    [
        ([], None, pytest.raises(ValueError)),
        (eddsa_public_key_json_payload, None, pytest.raises(ValueError)),
        ([another_algorithm_public_key_json_payload], None, pytest.raises(RuntimeError)),
        ([eddsa_public_key_json_payload], eddsa_public_key_pem, does_not_raise()),
    ],
)
def test_parse_jwt_public_key_json(payload: Any, expected_pem: str, expectation: Any) -> None:
    with expectation:
        pem = parse_jwt_public_key_json(algorithm=algorithm_eddsa, payload=payload)
        if expected_pem:
            assert pem == expected_pem


eddsa_public_key_pem_with_bad_linebreaks = (
    "-----BEGIN PUBLIC KEY-----\\nMCowBQYDK2VwAyEA+RBhgyNluwaIL5KFJb6ZOL2H1nmyI8mW4Z2EHGDGCXM=\\n-----END PUBLIC"
    " KEY-----"
)


@pytest.mark.parametrize(
    "payload,expected_pem,expectation",
    [
        (eddsa_public_key_pem_with_bad_linebreaks, None, pytest.raises(Exception)),
        (eddsa_public_key_pem, eddsa_public_key_pem, does_not_raise()),
    ],
)
def test_parse_jwt_public_key_pem(payload: Any, expected_pem: str, expectation: Any) -> None:
    with expectation:
        pem = parse_jwt_public_key_pem(algorithm=algorithm_eddsa, payload=payload)
        if expected_pem:
            assert pem == expected_pem


private_key_ok = SigningKey.generate(curve=Ed25519)
private_key_pem_ok = private_key_ok.to_pem(format="pkcs8")
public_key_pem_ok = private_key_ok.get_verifying_key().to_pem().decode("utf-8")

other_private_key = SigningKey.generate(curve=Ed25519)
other_private_key_pem = other_private_key.to_pem(format="pkcs8")
other_public_key_pem = other_private_key.get_verifying_key().to_pem().decode("utf-8")


@pytest.mark.parametrize(
    "keys_env_var,expected_keys",
    [
        ("", []),
        (public_key_pem_ok, [public_key_pem_ok]),
        (f"{public_key_pem_ok},{public_key_pem_ok}", [public_key_pem_ok, public_key_pem_ok]),
        (f"{public_key_pem_ok},{other_public_key_pem}", [public_key_pem_ok, other_public_key_pem]),
        (
            f"{public_key_pem_ok},{other_public_key_pem},{eddsa_public_key_pem}",
            [public_key_pem_ok, other_public_key_pem, eddsa_public_key_pem],
        ),
    ],
)
def test_get_jwt_public_keys_from_env(keys_env_var: str, expected_keys: list[str]) -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("API_HF_JWT_ADDITIONAL_PUBLIC_KEYS", keys_env_var)
    api_config = ApiConfig.from_env(hf_endpoint="")
    assert (
        get_jwt_public_keys(
            algorithm_name=algorithm_name_eddsa,
            additional_public_keys=api_config.hf_jwt_additional_public_keys,
        )
        == expected_keys
    )
    monkeypatch.undo()


@pytest.mark.parametrize(
    "remote_payload,keys_payload,expected_keys,expectation",
    [
        ([], [], None, pytest.raises(JWTKeysError)),
        ([another_algorithm_public_key_json_payload], [], None, pytest.raises(JWTKeysError)),
        (None, [eddsa_public_key_pem_with_bad_linebreaks], None, pytest.raises(JWTKeysError)),
        ([eddsa_public_key_json_payload], [], [eddsa_public_key_pem], does_not_raise()),
        (
            None,
            [public_key_pem_ok, other_public_key_pem, eddsa_public_key_pem],
            [public_key_pem_ok, other_public_key_pem, eddsa_public_key_pem],
            does_not_raise(),
        ),
        (
            [eddsa_public_key_json_payload],
            [public_key_pem_ok, other_public_key_pem, eddsa_public_key_pem],
            [eddsa_public_key_pem, public_key_pem_ok, other_public_key_pem, eddsa_public_key_pem],
            does_not_raise(),
        ),
    ],
)
def test_get_jwt_public_keys(
    remote_payload: Any, keys_payload: list[str], expected_keys: list[str], expectation: Any
) -> None:
    def fake_fetch(
        url: str,
        hf_timeout_seconds: Optional[float] = None,
    ) -> Any:
        return remote_payload

    with patch("libapi.jwt_token.fetch_jwt_public_key_json", wraps=fake_fetch):
        with expectation:
            keys = get_jwt_public_keys(
                algorithm_name=algorithm_name_eddsa,
                public_key_url=None if remote_payload is None else "mock",
                additional_public_keys=keys_payload,
            )
            if expected_keys:
                assert keys == expected_keys


token_for_severo_glue = (
    "eyJhbGciOiJFZERTQSJ9.eyJyZWFkIjp0cnVlLCJzdWIiOiJkYXRhc2V0cy9zZXZlcm8vZ2x1ZSIsImV4cCI6MTY3ODgwMjk0NH0"
    ".nIi1ZKinMBpYi4kKtirW-cQEt1cGnAziTGmJsZeN5UpE62jz4DcPaIPlSI5P5ciGOlTxy4SEhD1WITkQzpo3Aw"
)
dataset_severo_glue = "severo/glue"


def test_is_jwt_valid_with_ec() -> None:
    validate_jwt(
        dataset=dataset_severo_glue,
        token=token_for_severo_glue,
        public_keys=[eddsa_public_key_pem],
        algorithm=algorithm_name_eddsa,
        verify_exp=False,
        # This is a test token generated on 2023/03/14, so we don't want to verify the exp.
    )


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
algorithm_ok = algorithm_name_eddsa
algorithm_wrong = algorithm_name_rs256


def encode_jwt(payload: dict[str, Any]) -> str:
    return jwt.encode(payload, private_key_pem_ok, algorithm=algorithm_ok)


def assert_jwt(
    token: str, expectation: Any, public_keys: Optional[list[str]] = None, algorithm: str = algorithm_ok
) -> None:
    if public_keys is None:
        public_keys = [public_key_pem_ok]
    with expectation:
        validate_jwt(dataset=dataset_ok, token=token, public_keys=public_keys, algorithm=algorithm)


@pytest.mark.parametrize(
    "public_keys,expectation",
    [
        ([other_public_key_pem], pytest.raises(JWTInvalidSignature)),
        ([public_key_pem_ok], does_not_raise()),
        ([public_key_pem_ok, other_public_key_pem], does_not_raise()),
        ([other_public_key_pem, public_key_pem_ok], does_not_raise()),
    ],
)
def test_validate_jwt_public_keys(public_keys: list[str], expectation: Any) -> None:
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
def test_validate_jwt_content_format(payload: dict[str, str], expectation: Any) -> None:
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
