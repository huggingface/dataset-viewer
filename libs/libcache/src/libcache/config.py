from typing import Optional

from environs import Env


class CacheConfig:
    assets_directory: Optional[str]
    mongo_database: str
    mongo_url: str

    def __init__(self):
        env = Env(expand_vars=True)
        with env.prefixed("CACHE_"):
            self.assets_directory = env.str(name="ASSETS_DIRECTORY", default=None)
            self.mongo_database = env.str(name="MONGO_DATABASE", default="datasets_server_cache")
            self.mongo_url = env.str(name="MONGO_URL", default="mongodb://localhost:27017")
