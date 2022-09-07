import os

from dotenv import load_dotenv
from libutils.utils import get_int_value, get_str_or_none_value, get_str_value

from admin.constants import (
    DEFAULT_APP_HOSTNAME,
    DEFAULT_APP_NUM_WORKERS,
    DEFAULT_APP_PORT,
    DEFAULT_ASSETS_DIRECTORY,
    DEFAULT_CACHE_REPORTS_NUM_RESULTS,
    DEFAULT_HF_ENDPOINT,
    DEFAULT_HF_ORGANIZATION,
    DEFAULT_HF_WHOAMI_PATH,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_AGE_SHORT_SECONDS,
    DEFAULT_MONGO_CACHE_DATABASE,
    DEFAULT_MONGO_QUEUE_DATABASE,
    DEFAULT_MONGO_URL,
)

# Load environment variables defined in .env, if any
load_dotenv()

APP_HOSTNAME = get_str_value(d=os.environ, key="APP_HOSTNAME", default=DEFAULT_APP_HOSTNAME)
APP_NUM_WORKERS = get_int_value(d=os.environ, key="APP_NUM_WORKERS", default=DEFAULT_APP_NUM_WORKERS)
APP_PORT = get_int_value(d=os.environ, key="APP_PORT", default=DEFAULT_APP_PORT)
ASSETS_DIRECTORY = get_str_or_none_value(d=os.environ, key="ASSETS_DIRECTORY", default=DEFAULT_ASSETS_DIRECTORY)
CACHE_REPORTS_NUM_RESULTS = get_int_value(
    d=os.environ, key="CACHE_REPORTS_NUM_RESULTS", default=DEFAULT_CACHE_REPORTS_NUM_RESULTS
)
HF_ENDPOINT = get_str_value(d=os.environ, key="HF_ENDPOINT", default=DEFAULT_HF_ENDPOINT)
HF_ORGANIZATION = get_str_or_none_value(d=os.environ, key="HF_ORGANIZATION", default=DEFAULT_HF_ORGANIZATION)
HF_WHOAMI_PATH = get_str_or_none_value(d=os.environ, key="HF_WHOAMI_PATH", default=DEFAULT_HF_WHOAMI_PATH)
LOG_LEVEL = get_str_value(d=os.environ, key="LOG_LEVEL", default=DEFAULT_LOG_LEVEL)
MAX_AGE_SHORT_SECONDS = get_int_value(d=os.environ, key="MAX_AGE_SHORT_SECONDS", default=DEFAULT_MAX_AGE_SHORT_SECONDS)
MONGO_CACHE_DATABASE = get_str_value(d=os.environ, key="MONGO_CACHE_DATABASE", default=DEFAULT_MONGO_CACHE_DATABASE)
MONGO_QUEUE_DATABASE = get_str_value(d=os.environ, key="MONGO_QUEUE_DATABASE", default=DEFAULT_MONGO_QUEUE_DATABASE)
MONGO_URL = get_str_value(d=os.environ, key="MONGO_URL", default=DEFAULT_MONGO_URL)

EXTERNAL_AUTH_URL = None if HF_WHOAMI_PATH is None else f"{HF_ENDPOINT}{HF_WHOAMI_PATH}"
