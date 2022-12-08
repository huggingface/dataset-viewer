# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import importlib.metadata
import json
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Optional

import datasets.config
from libcommon.storage import init_dir, remove_dir
from libcommon.worker import Worker
from psutil import disk_usage

from datasets_based.config import AppConfig, DatasetsBasedConfig


class DatasetsBasedWorker(Worker, ABC):
    """Base class for workers that use datasets."""

    datasets_based_config: DatasetsBasedConfig

    @staticmethod
    @abstractmethod
    def get_endpoint() -> str:
        pass

    # the datasets library cache directories (for data, downloads, extraction, and modules)
    # the worker should have only one running job at the same time, then it should
    # be safe to use a global variable (and to set the datasets cache globally)
    datasets_cache: Optional[Path] = None
    modules_cache: Optional[Path] = None

    def __init__(self, app_config: AppConfig):
        super().__init__(
            processing_step=app_config.processing_graph.graph.get_step(self.get_endpoint()),
            # ^ raises if the step is not found
            common_config=app_config.common,
            queue_config=app_config.queue,
            worker_config=app_config.worker,
            version=importlib.metadata.version(__package__.split(".")[0]),
        )
        self.datasets_based_config = app_config.datasets_based

    def has_storage(self) -> bool:
        try:
            usage = disk_usage(str(self.datasets_based_config.hf_datasets_cache))
            return usage.percent < self.datasets_based_config.max_disk_usage_percent
        except Exception:
            # if we can't get the disk usage, we let the process continue
            return True

    def get_cache_subdirectory(
        self,
        date: datetime,
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
        force: bool = False,
    ) -> str:
        date_str = date.strftime("%Y-%m-%d-%H-%M-%S")
        payload = (date_str, self.get_endpoint(), dataset, config, split, force)
        hash_suffix = sha1(json.dumps(payload, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:8]
        prefix = f"{date_str}-{self.get_endpoint()}-{dataset}"[:64]
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
        self.set_datasets_cache(self.datasets_based_config.hf_datasets_cache)
        if previous_datasets_cache is not None and self.datasets_cache != previous_datasets_cache:
            remove_dir(previous_datasets_cache)
            logging.debug(f"temporary datasets data cache deleted: {previous_datasets_cache}")
        self.datasets_cache = None

    def set_modules_cache(self, modules_cache: Path) -> None:
        self.modules_cache = Path(init_dir(modules_cache))
        datasets.config.HF_MODULES_CACHE = self.modules_cache
        logging.debug(f"datasets modules cache set to: {datasets.config.HF_MODULES_CACHE}")

    def unset_modules_cache(self) -> None:
        previous_modules_cache = self.modules_cache
        self.set_modules_cache(self.datasets_based_config.hf_modules_cache)
        if previous_modules_cache is not None and self.modules_cache != previous_modules_cache:
            remove_dir(previous_modules_cache)
            logging.debug(f"temporary datasets modules cache deleted: {previous_modules_cache}")
        self.modules_cache = None

    def set_cache(
        self, dataset: str, config: Optional[str] = None, split: Optional[str] = None, force: bool = False
    ) -> None:
        cache_subdirectory = self.get_cache_subdirectory(
            date=datetime.now(), dataset=dataset, config=config, split=split, force=force
        )
        self.set_datasets_cache(self.datasets_based_config.hf_datasets_cache / cache_subdirectory)
        self.set_modules_cache(self.datasets_based_config.hf_modules_cache / cache_subdirectory)

    def unset_cache(self) -> None:
        self.unset_datasets_cache()
        self.unset_modules_cache()

    def pre_compute(
        self, dataset: str, config: Optional[str] = None, split: Optional[str] = None, force: bool = False
    ) -> None:
        self.set_cache(dataset=dataset, config=config, split=split, force=force)

    def post_compute(
        self, dataset: str, config: Optional[str] = None, split: Optional[str] = None, force: bool = False
    ) -> None:
        # empty the cache after the job to save storage space
        self.unset_cache()
