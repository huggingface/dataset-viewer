import os

from libutils.utils import get_int_value, get_str_value

DEFAULT_ASSETS_BASE_URL: str = "http://localhost/assets"
DEFAULT_HF_TOKEN: str = ""
DEFAULT_MONGO_CACHE_DATABASE: str = "datasets_server_cache_test"
DEFAULT_MONGO_QUEUE_DATABASE: str = "datasets_server_queue_test"
DEFAULT_MONGO_URL: str = "mongodb://localhost:27017"
DEFAULT_ROWS_MAX_NUMBER: int = 5

ASSETS_BASE_URL = get_str_value(d=os.environ, key="ASSETS_BASE_URL", default=DEFAULT_ASSETS_BASE_URL)
HF_TOKEN = get_str_value(d=os.environ, key="HF_TOKEN", default=DEFAULT_HF_TOKEN)
MONGO_CACHE_DATABASE = get_str_value(d=os.environ, key="MONGO_CACHE_DATABASE", default=DEFAULT_MONGO_CACHE_DATABASE)
MONGO_QUEUE_DATABASE = get_str_value(d=os.environ, key="MONGO_QUEUE_DATABASE", default=DEFAULT_MONGO_QUEUE_DATABASE)
MONGO_URL = get_str_value(d=os.environ, key="MONGO_URL", default=DEFAULT_MONGO_URL)
ROWS_MAX_NUMBER = get_int_value(d=os.environ, key="ROWS_MAX_NUMBER", default=DEFAULT_ROWS_MAX_NUMBER)
