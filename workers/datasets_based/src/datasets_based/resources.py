# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import datasets
from datasets.utils.logging import get_verbosity, log_levels, set_verbosity
from libcommon.resources import Resource


@dataclass
class LibrariesResource(Resource):
    hf_endpoint: str
    init_hf_datasets_cache: Optional[str] = None
    numba_path: Optional[str] = None

    previous_hf_endpoint: str = field(init=False)
    previous_hf_update_download_counts: bool = field(init=False)
    previous_verbosity: int = field(init=False)
    hf_datasets_cache: Path = field(init=False)
    storage_paths: set[str] = field(init=False)

    def allocate(self):
        self.hf_datasets_cache = (
            datasets.config.HF_DATASETS_CACHE
            if self.init_hf_datasets_cache is None
            else Path(self.init_hf_datasets_cache)
        )

        # Ensure the datasets library uses the expected HuggingFace endpoint
        self.previous_hf_endpoint = datasets.config.HF_ENDPOINT
        datasets.config.HF_ENDPOINT = self.hf_endpoint
        # Don't increase the datasets download counts on huggingface.co
        self.previous_hf_update_download_counts = datasets.config.HF_UPDATE_DOWNLOAD_COUNTS
        datasets.config.HF_UPDATE_DOWNLOAD_COUNTS = False
        # Set logs from the datasets library to the least verbose
        self.previous_verbosity = get_verbosity()
        set_verbosity(log_levels["critical"])

        # Note: self.hf_endpoint is ignored by the huggingface_hub library for now (see
        # the discussion at https://github.com/huggingface/datasets/pull/5196), and this breaks
        # various of the datasets functions. The fix, for now, is to set the HF_ENDPOINT
        # environment variable to the desired value.
        # TODO: check here if huggingface_hub and datasets use the same endpoint

        # Add the datasets and numba cache paths to the list of storage paths, to ensure the disk is not full
        storage_paths = {str(self.hf_datasets_cache), str(datasets.config.HF_MODULES_CACHE)}
        if self.numba_path is not None:
            storage_paths.add(self.numba_path)
        self.storage_paths = storage_paths

    def release(self):
        datasets.config.HF_ENDPOINT = self.previous_hf_endpoint
        datasets.config.HF_UPDATE_DOWNLOAD_COUNTS = self.previous_hf_update_download_counts
        set_verbosity(self.previous_verbosity)
