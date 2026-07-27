# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from environs import Env
from pytest import MonkeyPatch

from libcommon.config import S3Config
from libcommon.secrets import SecretsConfig, SecretsError, get_secrets, resolve_secret, resolve_secret_list


@pytest.fixture(autouse=True)
def clear_secrets_cache() -> Iterator[None]:
    get_secrets.cache_clear()
    yield
    get_secrets.cache_clear()


@pytest.fixture
def mount(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    monkeypatch.setenv("SECRETS_DIR", str(tmp_path))
    monkeypatch.setenv("SECRETS_TIMEOUT_SECONDS", "0")
    monkeypatch.setenv("SECRETS_POLL_INTERVAL_SECONDS", "0")
    return tmp_path


def test_disabled_by_default() -> None:
    assert not SecretsConfig.from_env().enabled
    assert get_secrets() == {}


def test_reads_one_file_per_secret(mount: Path) -> None:
    (mount / "HF_TOKEN").write_text("hf_xxx\n")
    (mount / "WEBHOOK_SECRET").write_text("s3cr3t")
    assert dict(get_secrets()) == {"HF_TOKEN": "hf_xxx", "WEBHOOK_SECRET": "s3cr3t"}


def test_reads_through_the_atomic_writer_layout(mount: Path) -> None:
    # The kubelet and the CSI driver keep versioned copies in ..data and symlink the real names next to
    # them, so the dot entries have to be skipped and the symlinks followed.
    data = mount / "..2026_07_28_00_00_00"
    data.mkdir()
    (data / "HF_TOKEN").write_text("hf_xxx")
    (mount / "..data").symlink_to(data)
    (mount / "HF_TOKEN").symlink_to(Path("..data") / "HF_TOKEN")

    assert dict(get_secrets()) == {"HF_TOKEN": "hf_xxx"}


def test_an_unmounted_directory_fails_loudly(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SECRETS_DIR", str(tmp_path / "absent"))
    with pytest.raises(SecretsError, match="not mounted"):
        get_secrets()


def test_an_empty_file_is_retried_then_reported(mount: Path) -> None:
    # A closed read window serves empty files, so empty means "not there yet" rather than absent. With
    # the timeout at zero this reports immediately instead of waiting.
    (mount / "HF_TOKEN").write_text("hf_xxx")
    (mount / "WEBHOOK_SECRET").write_text("")
    with pytest.raises(SecretsError, match="WEBHOOK_SECRET"):
        get_secrets()


def test_a_file_that_fills_in_late_is_picked_up(mount: Path, monkeypatch: MonkeyPatch) -> None:
    # Granting a read window takes a few seconds to take effect, so the first read has to wait it out.
    monkeypatch.setenv("SECRETS_TIMEOUT_SECONDS", "5")
    reads = iter([{"HF_TOKEN": ""}, {"HF_TOKEN": ""}, {"HF_TOKEN": "hf_xxx"}])
    with patch("libcommon.secrets._read_directory", side_effect=lambda _: next(reads)):
        assert dict(get_secrets()) == {"HF_TOKEN": "hf_xxx"}


def test_falls_back_to_the_environment(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("COMMON_HF_TOKEN", "token-from-env")
    env = Env(expand_vars=True)
    with env.prefixed("COMMON_"):
        assert resolve_secret(env, "HF_TOKEN", "HF_TOKEN", None) == "token-from-env"


def test_falls_back_to_the_default_when_unset() -> None:
    env = Env(expand_vars=True)
    with env.prefixed("COMMON_"):
        assert resolve_secret(env, "HF_TOKEN", "HF_TOKEN", None) is None
        assert resolve_secret(env, "MONGO_URL", "MONGO_URL", "mongodb://localhost:27017") == (
            "mongodb://localhost:27017"
        )


def test_the_mount_wins_over_the_environment(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("COMMON_HF_TOKEN", "token-from-env")
    env = Env(expand_vars=True)
    with patch("libcommon.secrets.get_secrets", return_value={"HF_TOKEN": "token-from-the-mount"}):
        with env.prefixed("COMMON_"):
            assert resolve_secret(env, "HF_TOKEN", "HF_TOKEN", None) == "token-from-the-mount"


def test_resolve_secret_list(monkeypatch: MonkeyPatch) -> None:
    env = Env(expand_vars=True)
    with env.prefixed("API_"):
        assert resolve_secret_list(env, "HF_JWT_ADDITIONAL_PUBLIC_KEYS", "KEYS", []) == []
        monkeypatch.setenv("API_HF_JWT_ADDITIONAL_PUBLIC_KEYS", "key1,key2")
        assert resolve_secret_list(env, "HF_JWT_ADDITIONAL_PUBLIC_KEYS", "KEYS", []) == ["key1", "key2"]
        with patch("libcommon.secrets.get_secrets", return_value={"KEYS": "key3, key4 ,"}):
            assert resolve_secret_list(env, "HF_JWT_ADDITIONAL_PUBLIC_KEYS", "KEYS", []) == ["key3", "key4"]


def test_irsa_keeps_the_static_s3_credentials_unresolved(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("S3_USE_IRSA", "true")
    with patch(
        "libcommon.secrets.get_secrets",
        return_value={"AWS_ACCESS_KEY_ID": "AKIA", "AWS_SECRET_ACCESS_KEY": "shhh"},
    ):
        s3_config = S3Config.from_env()
    assert s3_config.access_key_id is None
    assert s3_config.secret_access_key is None


def test_never_writes_the_secrets_to_the_environment(mount: Path) -> None:
    (mount / "WEBHOOK_SECRET").write_text("s3cr3t")
    assert get_secrets()["WEBHOOK_SECRET"] == "s3cr3t"  # nosec
    assert "WEBHOOK_SECRET" not in os.environ
    assert "s3cr3t" not in os.environ.values()
