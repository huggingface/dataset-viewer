# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os

import datasets.config
from datasets.utils.logging import log_levels, set_verbosity
from libutils.utils import get_int_value, get_str_or_none_value, get_str_value

from splits.constants import (
    DEFAULT_DATASETS_REVISION,
    DEFAULT_HF_ENDPOINT,
    DEFAULT_HF_TOKEN,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_JOBS_PER_DATASET,
    DEFAULT_MAX_LOAD_PCT,
    DEFAULT_MAX_MEMORY_PCT,
    DEFAULT_MONGO_CACHE_DATABASE,
    DEFAULT_MONGO_QUEUE_DATABASE,
    DEFAULT_MONGO_URL,
    DEFAULT_WORKER_SLEEP_SECONDS,
)

DATASETS_REVISION = get_str_value(d=os.environ, key="DATASETS_REVISION", default=DEFAULT_DATASETS_REVISION)
HF_ENDPOINT = get_str_value(d=os.environ, key="HF_ENDPOINT", default=DEFAULT_HF_ENDPOINT)
HF_TOKEN = get_str_or_none_value(d=os.environ, key="HF_TOKEN", default=DEFAULT_HF_TOKEN)
LOG_LEVEL = get_str_value(d=os.environ, key="LOG_LEVEL", default=DEFAULT_LOG_LEVEL)
MAX_JOBS_PER_DATASET = get_int_value(os.environ, "MAX_JOBS_PER_DATASET", DEFAULT_MAX_JOBS_PER_DATASET)
MAX_LOAD_PCT = get_int_value(os.environ, "MAX_LOAD_PCT", DEFAULT_MAX_LOAD_PCT)
MAX_MEMORY_PCT = get_int_value(os.environ, "MAX_MEMORY_PCT", DEFAULT_MAX_MEMORY_PCT)
MONGO_CACHE_DATABASE = get_str_value(d=os.environ, key="MONGO_CACHE_DATABASE", default=DEFAULT_MONGO_CACHE_DATABASE)
MONGO_QUEUE_DATABASE = get_str_value(d=os.environ, key="MONGO_QUEUE_DATABASE", default=DEFAULT_MONGO_QUEUE_DATABASE)
MONGO_URL = get_str_value(d=os.environ, key="MONGO_URL", default=DEFAULT_MONGO_URL)
WORKER_SLEEP_SECONDS = get_int_value(os.environ, "WORKER_SLEEP_SECONDS", DEFAULT_WORKER_SLEEP_SECONDS)

# Ensure the datasets library uses the expected revision for canonical datasets
# this one has to be set via an env variable unlike the others - this might be fixed in `datasets` at one point
os.environ["HF_SCRIPTS_VERSION"] = DATASETS_REVISION
# Ensure the datasets library uses the expected HuggingFace endpoint
datasets.config.HF_ENDPOINT = HF_ENDPOINT
# Don't increase the datasets download counts on huggingface.co
datasets.config.HF_UPDATE_DOWNLOAD_COUNTS = False
# Set logs from the datasets library to the least verbose
set_verbosity(log_levels["critical"])
