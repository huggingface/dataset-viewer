import os

from dotenv import load_dotenv
from libutils.utils import get_int_value, get_str_or_none_value, get_str_value

from job_runner.constants import (
    DEFAULT_HF_TOKEN,
    DEFAULT_MAX_JOBS_PER_DATASET,
    DEFAULT_MAX_LOAD_PCT,
    DEFAULT_MAX_MEMORY_PCT,
    DEFAULT_MAX_SIZE_FALLBACK,
    DEFAULT_ROWS_MAX_BYTES,
    DEFAULT_ROWS_MAX_NUMBER,
    DEFAULT_ROWS_MIN_NUMBER,
    DEFAULT_WORKER_QUEUE,
    DEFAULT_WORKER_SLEEP_SECONDS,
)

# Load environment variables defined in .env, if any
load_dotenv()

MAX_JOBS_PER_DATASET = get_int_value(os.environ, "MAX_JOBS_PER_DATASET", DEFAULT_MAX_JOBS_PER_DATASET)
MAX_LOAD_PCT = get_int_value(os.environ, "MAX_LOAD_PCT", DEFAULT_MAX_LOAD_PCT)
MAX_MEMORY_PCT = get_int_value(os.environ, "MAX_MEMORY_PCT", DEFAULT_MAX_MEMORY_PCT)
WORKER_SLEEP_SECONDS = get_int_value(os.environ, "WORKER_SLEEP_SECONDS", DEFAULT_WORKER_SLEEP_SECONDS)
HF_TOKEN = get_str_or_none_value(d=os.environ, key="HF_TOKEN", default=DEFAULT_HF_TOKEN)
MAX_SIZE_FALLBACK = get_int_value(os.environ, "MAX_SIZE_FALLBACK", DEFAULT_MAX_SIZE_FALLBACK)
ROWS_MAX_BYTES = get_int_value(os.environ, "ROWS_MAX_BYTES", DEFAULT_ROWS_MAX_BYTES)
ROWS_MAX_NUMBER = get_int_value(os.environ, "ROWS_MAX_NUMBER", DEFAULT_ROWS_MAX_NUMBER)
ROWS_MIN_NUMBER = get_int_value(os.environ, "ROWS_MIN_NUMBER", DEFAULT_ROWS_MIN_NUMBER)
WORKER_QUEUE = get_str_value(os.environ, "WORKER_QUEUE", DEFAULT_WORKER_QUEUE)
