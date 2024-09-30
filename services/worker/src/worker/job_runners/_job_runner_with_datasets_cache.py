# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from pathlib import Path
from typing import Optional

import datasets.config
import huggingface_hub.constants
from libcommon.dtos import JobInfo

from worker.config import AppConfig
from worker.job_runners._job_runner_with_cache import JobRunnerWithCache


class JobRunnerWithDatasetsCache(JobRunnerWithCache):
    """Base class for job runners that use datasets."""

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        hf_datasets_cache: Path,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            cache_directory=hf_datasets_cache,
        )

    def set_datasets_cache(self, cache_subdirectory: Optional[Path]) -> None:
        datasets.config.HF_DATASETS_CACHE = cache_subdirectory / "datasets"
        logging.debug(f"datasets data cache set to: {datasets.config.HF_DATASETS_CACHE}")
        datasets.config.DOWNLOADED_DATASETS_PATH = (
            datasets.config.HF_DATASETS_CACHE / datasets.config.DOWNLOADED_DATASETS_DIR
        )
        datasets.config.EXTRACTED_DATASETS_PATH = (
            datasets.config.HF_DATASETS_CACHE / datasets.config.EXTRACTED_DATASETS_DIR
        )
        huggingface_hub.constants.HF_HUB_CACHE = cache_subdirectory / "hub"
        logging.debug(f"huggingface_hub cache set to: {huggingface_hub.constants.HF_HUB_CACHE}")

    def pre_compute(self) -> None:
        super().pre_compute()
        self.set_datasets_cache(self.cache_subdirectory)

    def post_compute(self) -> None:
        super().post_compute()
        self.set_datasets_cache(self.base_cache_directory)
