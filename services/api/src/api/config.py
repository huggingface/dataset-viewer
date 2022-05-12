import os

from dotenv import load_dotenv
from libutils.utils import get_int_value, get_str_or_none_value, get_str_value

from api.constants import (
    DEFAULT_API_HOSTNAME,
    DEFAULT_API_NUM_WORKERS,
    DEFAULT_API_PORT,
    DEFAULT_ASSETS_DIRECTORY,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_AGE_LONG_SECONDS,
    DEFAULT_MAX_AGE_SHORT_SECONDS,
    DEFAULT_METRICS_HOSTNAME,
    DEFAULT_METRICS_NUM_WORKERS,
    DEFAULT_METRICS_PORT,
    DEFAULT_MONGO_CACHE_DATABASE,
    DEFAULT_MONGO_QUEUE_DATABASE,
    DEFAULT_MONGO_URL,
)

# Load environment variables defined in .env, if any
load_dotenv()

API_HOSTNAME = get_str_value(d=os.environ, key="API_HOSTNAME", default=DEFAULT_API_HOSTNAME)
API_NUM_WORKERS = get_int_value(d=os.environ, key="API_NUM_WORKERS", default=DEFAULT_API_NUM_WORKERS)
API_PORT = get_int_value(d=os.environ, key="API_PORT", default=DEFAULT_API_PORT)
ASSETS_DIRECTORY = get_str_or_none_value(d=os.environ, key="ASSETS_DIRECTORY", default=DEFAULT_ASSETS_DIRECTORY)
LOG_LEVEL = get_str_value(d=os.environ, key="LOG_LEVEL", default=DEFAULT_LOG_LEVEL)
MAX_AGE_LONG_SECONDS = get_int_value(d=os.environ, key="MAX_AGE_LONG_SECONDS", default=DEFAULT_MAX_AGE_LONG_SECONDS)
MAX_AGE_SHORT_SECONDS = get_int_value(d=os.environ, key="MAX_AGE_SHORT_SECONDS", default=DEFAULT_MAX_AGE_SHORT_SECONDS)
METRICS_HOSTNAME = get_str_value(d=os.environ, key="METRICS_HOSTNAME", default=DEFAULT_METRICS_HOSTNAME)
METRICS_NUM_WORKERS = get_int_value(d=os.environ, key="METRICS_NUM_WORKERS", default=DEFAULT_METRICS_NUM_WORKERS)
METRICS_PORT = get_int_value(d=os.environ, key="METRICS_PORT", default=DEFAULT_METRICS_PORT)
MONGO_CACHE_DATABASE = get_str_value(d=os.environ, key="MONGO_CACHE_DATABASE", default=DEFAULT_MONGO_CACHE_DATABASE)
MONGO_QUEUE_DATABASE = get_str_value(d=os.environ, key="MONGO_QUEUE_DATABASE", default=DEFAULT_MONGO_QUEUE_DATABASE)
MONGO_URL = get_str_value(d=os.environ, key="MONGO_URL", default=DEFAULT_MONGO_URL)
