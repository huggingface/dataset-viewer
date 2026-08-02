# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from datasets import config as _datasets_config

from libcommon import fsspec, packaged_modules  # noqa: F401 (configure fsspec, disable unsafe formats)

# This is just to make `datasets` faster:
# no need to check for a Parquet export since we will build it
_datasets_config.USE_PARQUET_EXPORT = False
