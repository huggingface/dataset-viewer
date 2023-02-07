# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import json
import logging
import re
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Optional

import datasets.config
from libcommon.processing_graph import ProcessingStep
from libcommon.storage import init_dir, remove_dir

from datasets_based.config import AppConfig, DatasetsBasedConfig
from datasets_based.worker import JobInfo, Worker


class DatasetsBasedWorker(Worker):
    """Base class for workers that use datasets."""

    datasets_based_config: DatasetsBasedConfig
    base_datasets_cache: Path

    # the datasets library cache directories (for data, downloads, extraction, NOT for modules)
    # the worker should have only one running job at the same time, then it should
    # be safe to use a global variable (and to set the datasets cache globally)
    datasets_cache: Optional[Path] = None

    def __init__(
        self, job_info: JobInfo, app_config: AppConfig, processing_step: ProcessingStep, hf_datasets_cache: Path
    ) -> None:
        super().__init__(job_info=job_info, common_config=app_config.common, processing_step=processing_step)
        self.datasets_based_config = app_config.datasets_based
        self.base_datasets_cache = hf_datasets_cache

    def get_cache_subdirectory(self, date: datetime) -> str:
        date_str = date.strftime("%Y-%m-%d-%H-%M-%S")
        payload = (date_str, self.get_job_type(), self.dataset, self.config, self.split, self.force)
        hash_suffix = sha1(json.dumps(payload, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:8]
        prefix = f"{date_str}-{self.get_job_type()}-{self.dataset}"[:64]
        subdirectory = f"{prefix}-{hash_suffix}"
        return "".join([c if re.match(r"[\w-]", c) else "-" for c in subdirectory])

    def set_datasets_cache(self, datasets_cache: Path) -> None:
        self.datasets_cache = Path(init_dir(datasets_cache))
        datasets.config.HF_DATASETS_CACHE = self.datasets_cache
        logging.debug(f"datasets data cache set to: {datasets.config.HF_DATASETS_CACHE}")
        datasets.config.DOWNLOADED_DATASETS_PATH = (
            datasets.config.HF_DATASETS_CACHE / datasets.config.DOWNLOADED_DATASETS_DIR
        )
        datasets.config.EXTRACTED_DATASETS_PATH = (
            datasets.config.HF_DATASETS_CACHE / datasets.config.EXTRACTED_DATASETS_DIR
        )

    def unset_datasets_cache(self) -> None:
        previous_datasets_cache = self.datasets_cache
        self.set_datasets_cache(self.base_datasets_cache)
        if previous_datasets_cache is not None and self.datasets_cache != previous_datasets_cache:
            remove_dir(previous_datasets_cache)
            logging.debug(f"temporary datasets data cache deleted: {previous_datasets_cache}")
        self.datasets_cache = None

    def set_cache(self) -> None:
        cache_subdirectory = self.get_cache_subdirectory(date=datetime.now())
        self.set_datasets_cache(self.base_datasets_cache / cache_subdirectory)

    def unset_cache(self) -> None:
        self.unset_datasets_cache()

    def pre_compute(self) -> None:
        self.set_cache()

    def post_compute(self) -> None:
        # empty the cache after the job to save storage space
        self.unset_cache()
