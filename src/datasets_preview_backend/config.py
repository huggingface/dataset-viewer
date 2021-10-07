import os

from dotenv import load_dotenv

from datasets_preview_backend.constants import (
    DEFAULT_APP_HOSTNAME,
    DEFAULT_APP_PORT,
    DEFAULT_CACHE_DIRECTORY,
    DEFAULT_CACHE_PERSIST,
    DEFAULT_CACHE_SHORT_TTL_SECONDS,
    DEFAULT_CACHE_SIZE_LIMIT,
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_DATASETS_ENABLE_PRIVATE,
    DEFAULT_EXTRACT_ROWS_LIMIT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_WEB_CONCURRENCY,
)
from datasets_preview_backend.utils import (
    get_bool_value,
    get_int_value,
    get_str_or_none_value,
    get_str_value,
)

# Load environment variables defined in .env, if any
load_dotenv()

APP_HOSTNAME = os.environ.get("APP_HOSTNAME", DEFAULT_APP_HOSTNAME)
APP_PORT = get_int_value(d=os.environ, key="APP_PORT", default=DEFAULT_APP_PORT)
CACHE_DIRECTORY = get_str_or_none_value(d=os.environ, key="CACHE_DIRECTORY", default=DEFAULT_CACHE_DIRECTORY)
CACHE_PERSIST = get_bool_value(d=os.environ, key="CACHE_PERSIST", default=DEFAULT_CACHE_PERSIST)
CACHE_SHORT_TTL_SECONDS = get_int_value(
    d=os.environ, key="CACHE_SHORT_TTL_SECONDS", default=DEFAULT_CACHE_SHORT_TTL_SECONDS
)
CACHE_SIZE_LIMIT = get_int_value(d=os.environ, key="CACHE_SIZE_LIMIT", default=DEFAULT_CACHE_SIZE_LIMIT)
CACHE_TTL_SECONDS = get_int_value(d=os.environ, key="CACHE_TTL_SECONDS", default=DEFAULT_CACHE_TTL_SECONDS)
DATASETS_ENABLE_PRIVATE = get_bool_value(
    d=os.environ, key="DATASETS_ENABLE_PRIVATE", default=DEFAULT_DATASETS_ENABLE_PRIVATE
)
EXTRACT_ROWS_LIMIT = get_int_value(d=os.environ, key="EXTRACT_ROWS_LIMIT", default=DEFAULT_EXTRACT_ROWS_LIMIT)
LOG_LEVEL = get_str_value(d=os.environ, key="LOG_LEVEL", default=DEFAULT_LOG_LEVEL)
WEB_CONCURRENCY = get_int_value(d=os.environ, key="WEB_CONCURRENCY", default=DEFAULT_WEB_CONCURRENCY)
