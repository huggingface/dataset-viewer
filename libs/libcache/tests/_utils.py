import os

from libutils.utils import get_str_value

DEFAULT_MONGO_CACHE_DATABASE: str = "datasets_server_cache_test"
DEFAULT_MONGO_URL: str = "mongodb://localhost:27017"

MONGO_CACHE_DATABASE = get_str_value(d=os.environ, key="MONGO_CACHE_DATABASE", default=DEFAULT_MONGO_CACHE_DATABASE)
MONGO_URL = get_str_value(d=os.environ, key="MONGO_URL", default=DEFAULT_MONGO_URL)
