# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from pathlib import Path
from typing import List, Optional

import datasets.config
from datasets.utils.logging import log_levels, set_verbosity
from environs import Env
from libcommon.config import (
    AssetsConfig,
    CacheConfig,
    CommonConfig,
    ProcessingGraphConfig,
    QueueConfig,
    WorkerConfig,
)


class DatasetsBasedConfig:
    endpoint: str
    hf_datasets_cache: Path
    hf_modules_cache: Path
    max_disk_usage_percent = 90  # hard-coded, not configurable

    def __init__(self):
        env = Env(expand_vars=True)
        with env.prefixed("DATASETS_BASED_"):
            self.endpoint = env.str(name="ENDPOINT", default="/splits")
            self._hf_datasets_cache = env.str(name="HF_DATASETS_CACHE", default=None)
            self._hf_modules_cache = env.str(name="HF_MODULES_CACHE", default=None)
        self.setup()

    def setup(self) -> None:
        self.hf_datasets_cache = (
            datasets.config.HF_DATASETS_CACHE if self._hf_datasets_cache is None else Path(self._hf_datasets_cache)
        )
        self.hf_modules_cache = (
            datasets.config.HF_MODULES_CACHE if self._hf_modules_cache is None else Path(self._hf_modules_cache)
        )


class FirstRowsConfig:
    assets: AssetsConfig
    fallback_max_dataset_size: int
    max_bytes: int
    max_number: int
    min_cell_bytes: int
    min_number: int

    def __init__(self):
        env = Env(expand_vars=True)
        with env.prefixed("FIRST_ROWS_"):
            self.fallback_max_dataset_size = env.int(name="FALLBACK_MAX_DATASET_SIZE", default=100_000_000)
            self.max_bytes = env.int(name="MAX_BYTES", default=1_000_000)
            self.max_number = env.int(name="MAX_NUMBER", default=100)
            self.min_cell_bytes = env.int(name="CELL_MIN_BYTES", default=100)
            self.min_number = env.int(name="MIN_NUMBER", default=10)
        self.assets = AssetsConfig()


class ParquetConfig:
    blocked_datasets: List[str]
    supported_datasets: List[str]
    commit_message: str
    committer_hf_token: Optional[str]
    max_dataset_size: int
    source_revision: str
    target_revision: str
    url_template: str

    def __init__(self):
        env = Env(expand_vars=True)
        with env.prefixed("PARQUET_"):
            self.blocked_datasets = env.list(name="BLOCKED_DATASETS", default=[])
            self.supported_datasets = env.list(name="SUPPORTED_DATASETS", default=[])
            self.commit_message = env.str(name="COMMIT_MESSAGE", default="Update parquet files")
            self.committer_hf_token = env.str(name="COMMITTER_HF_TOKEN", default=None)
            self.max_dataset_size = env.int(name="MAX_DATASET_SIZE", default=100_000_000)
            self.source_revision = env.str(name="SOURCE_REVISION", default="main")
            self.target_revision = env.str(name="TARGET_REVISION", default="refs/convert/parquet")
            self.url_template = env.str(name="URL_TEMPLATE", default="/datasets/%s/resolve/%s/%s")


class AppConfig:
    cache: CacheConfig
    common: CommonConfig
    datasets_based: DatasetsBasedConfig
    processing_graph: ProcessingGraphConfig
    queue: QueueConfig
    worker: WorkerConfig

    def __init__(self):
        # First process the common configuration to setup the logging
        self.common = CommonConfig()
        self.cache = CacheConfig()
        self.datasets_based = DatasetsBasedConfig()
        self.processing_graph = ProcessingGraphConfig()
        self.queue = QueueConfig()
        self.worker = WorkerConfig()
        self.setup()

    def setup(self):
        # Ensure the datasets library uses the expected HuggingFace endpoint
        datasets.config.HF_ENDPOINT = self.common.hf_endpoint
        # Don't increase the datasets download counts on huggingface.co
        datasets.config.HF_UPDATE_DOWNLOAD_COUNTS = False
        # Set logs from the datasets library to the least verbose
        set_verbosity(log_levels["critical"])

        # Note: self.common.hf_endpoint is ignored by the huggingface_hub library for now (see
        # the discussion at https://github.com/huggingface/datasets/pull/5196), and this breaks
        # various of the datasets functions. The fix, for now, is to set the HF_ENDPOINT
        # environment variable to the desired value.
