# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field
from pathlib import Path

import datasets.config
from datasets.utils.logging import log_levels, set_verbosity
from libcommon.config import CommonConfig
from libcommon.resource import Resource

from datasets_based.config import DatasetsBasedConfig, NumbaConfig


@dataclass
class LibrariesResource(Resource):
    common_config: CommonConfig
    datasets_based_config: DatasetsBasedConfig
    numba_config: NumbaConfig

    hf_datasets_cache: Path = field(init=False)
    storage_paths: set[str] = field(init=False)

    def allocate(self):
        self.hf_datasets_cache = (
            datasets.config.HF_DATASETS_CACHE
            if self.datasets_based_config.hf_datasets_cache is None
            else Path(self.datasets_based_config.hf_datasets_cache)
        )

        # Ensure the datasets library uses the expected HuggingFace endpoint
        datasets.config.HF_ENDPOINT = self.common_config.hf_endpoint
        # Don't increase the datasets download counts on huggingface.co
        datasets.config.HF_UPDATE_DOWNLOAD_COUNTS = False
        # Set logs from the datasets library to the least verbose
        set_verbosity(log_levels["critical"])

        # Note: self.common_config.hf_endpoint is ignored by the huggingface_hub library for now (see
        # the discussion at https://github.com/huggingface/datasets/pull/5196), and this breaks
        # various of the datasets functions. The fix, for now, is to set the HF_ENDPOINT
        # environment variable to the desired value.
        # TODO: check here if huggingface_hub and datasets use the same endpoint

        # Add the datasets and numba cache paths to the list of storage paths, to ensure the disk is not full
        storage_paths = {str(self.hf_datasets_cache), str(datasets.config.HF_MODULES_CACHE)}
        if self.numba_config.path is not None:
            storage_paths.add(self.numba_config.path)
        self.storage_paths = storage_paths
