# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Optional

from environs import Env

from libcache.asset import init_assets_dir
from libcache.simple_cache import connect_to_database


class CacheConfig:
    _assets_directory: Optional[str]
    assets_directory: str
    mongo_database: str
    mongo_url: str

    def __init__(self):
        env = Env(expand_vars=True)
        with env.prefixed("CACHE_"):
            self._assets_directory = env.str(name="ASSETS_DIRECTORY", default=None)
            self.mongo_database = env.str(name="MONGO_DATABASE", default="datasets_server_cache")
            self.mongo_url = env.str(name="MONGO_URL", default="mongodb://localhost:27017")
        self.setup()

    def setup(self):
        connect_to_database(database=self.mongo_database, host=self.mongo_url)
        self.assets_directory = init_assets_dir(assets_directory=self._assets_directory)
