import os

from datasets_preview_backend.utils import (
    get_bool_value,
    get_int_value,
    get_str_or_none_value,
    get_str_value,
)
from dotenv import load_dotenv

from api_service.constants import (
    DEFAULT_APP_HOSTNAME,
    DEFAULT_APP_PORT,
    DEFAULT_ASSETS_DIRECTORY,
    DEFAULT_DATASETS_ENABLE_PRIVATE,
    DEFAULT_DATASETS_REVISION,
    DEFAULT_HF_TOKEN,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_AGE_LONG_SECONDS,
    DEFAULT_MAX_AGE_SHORT_SECONDS,
    DEFAULT_MAX_SIZE_FALLBACK,
    DEFAULT_MONGO_CACHE_DATABASE,
    DEFAULT_MONGO_QUEUE_DATABASE,
    DEFAULT_MONGO_URL,
    DEFAULT_ROWS_MAX_BYTES,
    DEFAULT_ROWS_MAX_NUMBER,
    DEFAULT_ROWS_MIN_NUMBER,
    DEFAULT_WEB_CONCURRENCY,
)

# Load environment variables defined in .env, if any
load_dotenv()

APP_HOSTNAME = get_str_value(d=os.environ, key="APP_HOSTNAME", default=DEFAULT_APP_HOSTNAME)
APP_PORT = get_int_value(d=os.environ, key="APP_PORT", default=DEFAULT_APP_PORT)
ASSETS_DIRECTORY = get_str_or_none_value(d=os.environ, key="ASSETS_DIRECTORY", default=DEFAULT_ASSETS_DIRECTORY)
DATASETS_ENABLE_PRIVATE = get_bool_value(
    d=os.environ, key="DATASETS_ENABLE_PRIVATE", default=DEFAULT_DATASETS_ENABLE_PRIVATE
)
DATASETS_REVISION = get_str_value(d=os.environ, key="DATASETS_REVISION", default=DEFAULT_DATASETS_REVISION)
HF_TOKEN = get_str_or_none_value(d=os.environ, key="HF_TOKEN", default=DEFAULT_HF_TOKEN)
LOG_LEVEL = get_str_value(d=os.environ, key="LOG_LEVEL", default=DEFAULT_LOG_LEVEL)
MAX_AGE_LONG_SECONDS = get_int_value(d=os.environ, key="MAX_AGE_LONG_SECONDS", default=DEFAULT_MAX_AGE_LONG_SECONDS)
MAX_AGE_SHORT_SECONDS = get_int_value(d=os.environ, key="MAX_AGE_SHORT_SECONDS", default=DEFAULT_MAX_AGE_SHORT_SECONDS)
MONGO_CACHE_DATABASE = get_str_value(d=os.environ, key="MONGO_CACHE_DATABASE", default=DEFAULT_MONGO_CACHE_DATABASE)
MONGO_QUEUE_DATABASE = get_str_value(d=os.environ, key="MONGO_QUEUE_DATABASE", default=DEFAULT_MONGO_QUEUE_DATABASE)
MONGO_URL = get_str_value(d=os.environ, key="MONGO_URL", default=DEFAULT_MONGO_URL)
WEB_CONCURRENCY = get_int_value(d=os.environ, key="WEB_CONCURRENCY", default=DEFAULT_WEB_CONCURRENCY)

# Ensure datasets library uses the expected revision for canonical datasets
os.environ["HF_SCRIPTS_VERSION"] = DATASETS_REVISION

# for tests - to be removed
MAX_SIZE_FALLBACK = get_int_value(os.environ, "MAX_SIZE_FALLBACK", DEFAULT_MAX_SIZE_FALLBACK)
ROWS_MAX_BYTES = get_int_value(d=os.environ, key="ROWS_MAX_BYTES", default=DEFAULT_ROWS_MAX_BYTES)
ROWS_MAX_NUMBER = get_int_value(d=os.environ, key="ROWS_MAX_NUMBER", default=DEFAULT_ROWS_MAX_NUMBER)
ROWS_MIN_NUMBER = get_int_value(d=os.environ, key="ROWS_MIN_NUMBER", default=DEFAULT_ROWS_MIN_NUMBER)
