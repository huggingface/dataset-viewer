# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os

DEFAULT_MONGO_CACHE_DATABASE: str = "datasets_server_cache_test"
DEFAULT_MONGO_URL: str = "mongodb://localhost:27017"


def get_str_value(d: os._Environ[str], key: str, default: str) -> str:
    if key not in d:
        return default
    value = str(d.get(key)).strip()
    return value or default


MONGO_CACHE_DATABASE = get_str_value(d=os.environ, key="MONGO_CACHE_DATABASE", default=DEFAULT_MONGO_CACHE_DATABASE)
MONGO_URL = get_str_value(d=os.environ, key="MONGO_URL", default=DEFAULT_MONGO_URL)
