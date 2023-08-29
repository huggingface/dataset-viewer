# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import json
import random
import re
from hashlib import sha1
from pathlib import Path
from typing import Optional

from libcommon.exceptions import DiskError
from libcommon.processing_graph import ProcessingStep
from libcommon.storage import init_dir, remove_dir
from libcommon.utils import JobInfo

from worker.config import AppConfig
from worker.job_runner import JobRunner


class JobRunnerWithCache(JobRunner):
    """Base class for job runners that use a temporary cache directory."""

    base_cache_directory: Path
    cache_subdirectory: Optional[Path] = None

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
        cache_directory: Path,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
        )
        self.base_cache_directory = cache_directory

    def get_cache_subdirectory(self, digits: int = 14) -> str:
        random_str = f"{random.randrange(10**(digits - 1), 10**digits)}"  # nosec B311
        # TODO: Refactor, need a way to generate payload based only on provided params
        payload = (
            random_str,
            self.get_job_type(),
            self.job_info["params"]["dataset"],
            self.job_info["params"]["config"],
            self.job_info["params"]["split"],
        )
        hash_suffix = sha1(json.dumps(payload, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:8]
        prefix = f"{random_str}-{self.get_job_type()}-{self.job_info['params']['dataset']}"[:64]
        subdirectory = f"{prefix}-{hash_suffix}"
        return "".join([c if re.match(r"[\w-]", c) else "-" for c in subdirectory])

    def pre_compute(self) -> None:
        new_directory = self.base_cache_directory / self.get_cache_subdirectory()
        try:
            self.cache_subdirectory = Path(init_dir(new_directory))
        except PermissionError as e:
            raise DiskError(f"Incorrect permissions on {new_directory}", e) from e

    def post_compute(self) -> None:
        # empty the cache after the job to save storage space
        previous_cache = self.cache_subdirectory
        if previous_cache is not None:
            remove_dir(previous_cache)
        self.cache_subdirectory = None
