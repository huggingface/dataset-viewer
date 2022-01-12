import os
import sys

from dotenv import load_dotenv

from datasets_preview_backend.constants import (
    DEFAULT_HF_TOKEN,
    DEFAULT_MAX_SIZE_FALLBACK,
)
from datasets_preview_backend.io.cache import connect_to_cache, refresh_dataset
from datasets_preview_backend.io.logger import init_logger
from datasets_preview_backend.utils import get_int_value, get_str_or_none_value

# Load environment variables defined in .env, if any
load_dotenv()
hf_token = get_str_or_none_value(d=os.environ, key="HF_TOKEN", default=DEFAULT_HF_TOKEN)
max_size_fallback = get_int_value(os.environ, "MAX_SIZE_FALLBACK", DEFAULT_MAX_SIZE_FALLBACK)

if __name__ == "__main__":
    init_logger("DEBUG")
    connect_to_cache()
    dataset_name = sys.argv[1]
    refresh_dataset(dataset_name=dataset_name, hf_token=hf_token, max_size_fallback=max_size_fallback)
