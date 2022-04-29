import os

from dotenv import load_dotenv
from libutils.utils import get_str_or_none_value, get_str_value

from libcache.constants import (
    DEFAULT_ASSETS_DIRECTORY,
    DEFAULT_MONGO_CACHE_DATABASE,
    DEFAULT_MONGO_URL,
)

# Load environment variables defined in .env, if any
load_dotenv()

ASSETS_DIRECTORY = get_str_or_none_value(d=os.environ, key="ASSETS_DIRECTORY", default=DEFAULT_ASSETS_DIRECTORY)
MONGO_CACHE_DATABASE = get_str_value(d=os.environ, key="MONGO_CACHE_DATABASE", default=DEFAULT_MONGO_CACHE_DATABASE)
MONGO_URL = get_str_value(d=os.environ, key="MONGO_URL", default=DEFAULT_MONGO_URL)
