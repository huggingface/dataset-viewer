# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from datasets import config as _datasets_config
from libcommon.ssrf import install_ssrf_guard as _install_ssrf_guard

# This is just to make `datasets` faster:
# no need to check for a Parquet export since we will build it
_datasets_config.USE_PARQUET_EXPORT = False

# The worker fetches URLs that come from dataset content, so it must never be able to reach an
# internal address. Installed here to be sure it is done before the first request.
_install_ssrf_guard()
