# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Optional

DEFAULT_HF_ENDPOINT: str = "https://huggingface.co"
DEFAULT_HF_TOKEN: Optional[str] = None
DEFAULT_LOG_LEVEL: str = "INFO"
DEFAULT_MAX_JOBS_PER_DATASET: int = 1
DEFAULT_MAX_LOAD_PCT: int = 70
DEFAULT_MAX_MEMORY_PCT: int = 80
DEFAULT_MONGO_CACHE_DATABASE: str = "datasets_server_cache"
DEFAULT_MONGO_QUEUE_DATABASE: str = "datasets_server_queue"
DEFAULT_MONGO_URL: str = "mongodb://localhost:27018"
DEFAULT_WORKER_SLEEP_SECONDS: int = 15
DEFAULT_WORKER_QUEUE: str = "splits_responses"
