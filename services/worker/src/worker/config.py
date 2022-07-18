import os

from datasets.utils.logging import log_levels, set_verbosity
from dotenv import load_dotenv
from libutils.utils import get_int_value, get_str_or_none_value, get_str_value

from worker.constants import (
    DEFAULT_ASSETS_BASE_URL,
    DEFAULT_ASSETS_DIRECTORY,
    DEFAULT_DATASETS_REVISION,
    DEFAULT_HF_TOKEN,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_JOB_RETRIES,
    DEFAULT_MAX_JOBS_PER_DATASET,
    DEFAULT_MAX_LOAD_PCT,
    DEFAULT_MAX_MEMORY_PCT,
    DEFAULT_MAX_SIZE_FALLBACK,
    DEFAULT_MIN_CELL_BYTES,
    DEFAULT_MONGO_CACHE_DATABASE,
    DEFAULT_MONGO_QUEUE_DATABASE,
    DEFAULT_MONGO_URL,
    DEFAULT_ROWS_MAX_BYTES,
    DEFAULT_ROWS_MAX_NUMBER,
    DEFAULT_ROWS_MIN_NUMBER,
    DEFAULT_WORKER_QUEUE,
    DEFAULT_WORKER_SLEEP_SECONDS,
)

# Load environment variables defined in .env, if any
load_dotenv()

ASSETS_BASE_URL = get_str_value(d=os.environ, key="ASSETS_BASE_URL", default=DEFAULT_ASSETS_BASE_URL)
ASSETS_DIRECTORY = get_str_or_none_value(d=os.environ, key="ASSETS_DIRECTORY", default=DEFAULT_ASSETS_DIRECTORY)
DATASETS_REVISION = get_str_value(d=os.environ, key="DATASETS_REVISION", default=DEFAULT_DATASETS_REVISION)
HF_TOKEN = get_str_or_none_value(d=os.environ, key="HF_TOKEN", default=DEFAULT_HF_TOKEN)
LOG_LEVEL = get_str_value(d=os.environ, key="LOG_LEVEL", default=DEFAULT_LOG_LEVEL)
MAX_JOB_RETRIES = get_int_value(os.environ, "MAX_JOB_RETRIES", DEFAULT_MAX_JOB_RETRIES)
MAX_JOBS_PER_DATASET = get_int_value(os.environ, "MAX_JOBS_PER_DATASET", DEFAULT_MAX_JOBS_PER_DATASET)
MAX_LOAD_PCT = get_int_value(os.environ, "MAX_LOAD_PCT", DEFAULT_MAX_LOAD_PCT)
MAX_MEMORY_PCT = get_int_value(os.environ, "MAX_MEMORY_PCT", DEFAULT_MAX_MEMORY_PCT)
MAX_SIZE_FALLBACK = get_int_value(os.environ, "MAX_SIZE_FALLBACK", DEFAULT_MAX_SIZE_FALLBACK)
MIN_CELL_BYTES = get_int_value(os.environ, "MIN_CELL_BYTES", DEFAULT_MIN_CELL_BYTES)
MONGO_CACHE_DATABASE = get_str_value(d=os.environ, key="MONGO_CACHE_DATABASE", default=DEFAULT_MONGO_CACHE_DATABASE)
MONGO_QUEUE_DATABASE = get_str_value(d=os.environ, key="MONGO_QUEUE_DATABASE", default=DEFAULT_MONGO_QUEUE_DATABASE)
MONGO_URL = get_str_value(d=os.environ, key="MONGO_URL", default=DEFAULT_MONGO_URL)
ROWS_MAX_BYTES = get_int_value(os.environ, "ROWS_MAX_BYTES", DEFAULT_ROWS_MAX_BYTES)
ROWS_MAX_NUMBER = get_int_value(os.environ, "ROWS_MAX_NUMBER", DEFAULT_ROWS_MAX_NUMBER)
ROWS_MIN_NUMBER = get_int_value(os.environ, "ROWS_MIN_NUMBER", DEFAULT_ROWS_MIN_NUMBER)
WORKER_QUEUE = get_str_value(os.environ, "WORKER_QUEUE", DEFAULT_WORKER_QUEUE)
WORKER_SLEEP_SECONDS = get_int_value(os.environ, "WORKER_SLEEP_SECONDS", DEFAULT_WORKER_SLEEP_SECONDS)

# Ensure the datasets library uses the expected revision for canonical datasets
os.environ["HF_SCRIPTS_VERSION"] = DATASETS_REVISION
# Set logs from the datasets library to the least verbose
set_verbosity(log_levels["critical"])
