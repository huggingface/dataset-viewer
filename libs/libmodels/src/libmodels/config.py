import os

from dotenv import load_dotenv
from libutils.utils import get_int_value, get_str_or_none_value, get_str_value

from libmodels.constants import (
    DEFAULT_ASSETS_DIRECTORY,
    DEFAULT_DATASETS_REVISION,
    DEFAULT_HF_TOKEN,
    DEFAULT_MAX_SIZE_FALLBACK,
    DEFAULT_ROWS_MAX_NUMBER,
)

# Load environment variables defined in .env, if any
load_dotenv()

ASSETS_DIRECTORY = get_str_or_none_value(d=os.environ, key="ASSETS_DIRECTORY", default=DEFAULT_ASSETS_DIRECTORY)
DATASETS_REVISION = get_str_value(d=os.environ, key="DATASETS_REVISION", default=DEFAULT_DATASETS_REVISION)
HF_TOKEN = get_str_or_none_value(d=os.environ, key="HF_TOKEN", default=DEFAULT_HF_TOKEN)
MAX_SIZE_FALLBACK = get_int_value(os.environ, "MAX_SIZE_FALLBACK", DEFAULT_MAX_SIZE_FALLBACK)
ROWS_MAX_NUMBER = get_int_value(d=os.environ, key="ROWS_MAX_NUMBER", default=DEFAULT_ROWS_MAX_NUMBER)

# Ensure datasets library uses the expected revision for canonical datasets
os.environ["HF_SCRIPTS_VERSION"] = DATASETS_REVISION
