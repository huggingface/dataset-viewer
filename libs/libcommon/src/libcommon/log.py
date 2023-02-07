# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging


def init_logging(log_level: int = logging.INFO) -> int:
    logging.basicConfig(level=log_level, format="%(levelname)s: %(asctime)s - %(name)s - %(message)s")
    logging.debug(f"Log level set to: {logging.getLevelName(logging.getLogger().getEffectiveLevel())}")
    return log_level
