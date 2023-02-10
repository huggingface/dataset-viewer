# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from environs import Env
from pytest import fixture


@fixture(scope="session")
def env() -> Env:
    return Env(expand_vars=True)


@fixture(scope="session")
def mongo_host(env: Env) -> str:
    try:
        return env.str(name="DATABASE_MIGRATIONS_MONGO_URL")
    except Exception as e:
        raise ValueError("DATABASE_MIGRATIONS_MONGO_URL is not set") from e
