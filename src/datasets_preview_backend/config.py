import os

from dotenv import load_dotenv

from datasets_preview_backend.constants import (
    DEFAULT_APP_HOSTNAME,
    DEFAULT_APP_PORT,
    DEFAULT_ASSETS_DIRECTORY,
    DEFAULT_CACHE_DIRECTORY,
    DEFAULT_CACHE_PERSIST,
    DEFAULT_CACHE_SIZE_LIMIT,
    DEFAULT_DATASETS_ENABLE_PRIVATE,
    DEFAULT_DATASETS_REVISION,
    DEFAULT_EXTRACT_ROWS_LIMIT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_AGE_LONG_SECONDS,
    DEFAULT_MAX_AGE_SHORT_SECONDS,
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
ASSETS_DIRECTORY = get_str_or_none_value(d=os.environ, key="ASSETS_DIRECTORY", default=DEFAULT_ASSETS_DIRECTORY)
CACHE_DIRECTORY = get_str_or_none_value(d=os.environ, key="CACHE_DIRECTORY", default=DEFAULT_CACHE_DIRECTORY)
CACHE_PERSIST = get_bool_value(d=os.environ, key="CACHE_PERSIST", default=DEFAULT_CACHE_PERSIST)
CACHE_SIZE_LIMIT = get_int_value(d=os.environ, key="CACHE_SIZE_LIMIT", default=DEFAULT_CACHE_SIZE_LIMIT)
DATASETS_ENABLE_PRIVATE = get_bool_value(
    d=os.environ, key="DATASETS_ENABLE_PRIVATE", default=DEFAULT_DATASETS_ENABLE_PRIVATE
)
DATASETS_REVISION = get_str_value(d=os.environ, key="DATASETS_REVISION", default=DEFAULT_DATASETS_REVISION)
EXTRACT_ROWS_LIMIT = get_int_value(d=os.environ, key="EXTRACT_ROWS_LIMIT", default=DEFAULT_EXTRACT_ROWS_LIMIT)
LOG_LEVEL = get_str_value(d=os.environ, key="LOG_LEVEL", default=DEFAULT_LOG_LEVEL)
MAX_AGE_LONG_SECONDS = get_int_value(d=os.environ, key="MAX_AGE_LONG_SECONDS", default=DEFAULT_MAX_AGE_LONG_SECONDS)
MAX_AGE_SHORT_SECONDS = get_int_value(d=os.environ, key="MAX_AGE_SHORT_SECONDS", default=DEFAULT_MAX_AGE_SHORT_SECONDS)
WEB_CONCURRENCY = get_int_value(d=os.environ, key="WEB_CONCURRENCY", default=DEFAULT_WEB_CONCURRENCY)

# Ensure datasets library uses the excepted revision
os.environ["HF_SCRIPTS_VERSION"] = DATASETS_REVISION
