# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from pathlib import Path
from typing import Optional

import datasets.config
from libcommon.processing_graph import ProcessingStep
from libcommon.utils import JobInfo

from worker.config import AppConfig
from worker.job_runners._cached_based_job_runner import CachedBasedJobRunner


class DatasetsBasedJobRunner(CachedBasedJobRunner):
    """Base class for job runners that use datasets."""

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
        hf_datasets_cache: Path,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
            cache_directory=hf_datasets_cache,
        )

    def set_datasets_cache(self, cache_subdirectory: Optional[Path]) -> None:
        datasets.config.HF_DATASETS_CACHE = cache_subdirectory
        logging.debug(f"datasets data cache set to: {datasets.config.HF_DATASETS_CACHE}")
        datasets.config.DOWNLOADED_DATASETS_PATH = (
            datasets.config.HF_DATASETS_CACHE / datasets.config.DOWNLOADED_DATASETS_DIR
        )
        datasets.config.EXTRACTED_DATASETS_PATH = (
            datasets.config.HF_DATASETS_CACHE / datasets.config.EXTRACTED_DATASETS_DIR
        )

    def pre_compute(self) -> None:
        super().pre_compute()
        self.set_datasets_cache(self.cache_subdirectory)

    def post_compute(self) -> None:
        super().post_compute()
        self.set_datasets_cache(self.base_cache_directory)
