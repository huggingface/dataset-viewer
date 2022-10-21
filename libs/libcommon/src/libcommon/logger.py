# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging


def init_logger(log_level: int = logging.INFO, name: str = "datasets_server") -> None:
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    formatter = logging.Formatter("%(levelname)s: %(asctime)s - %(name)s - %(message)s")

    logHandler = logging.StreamHandler()
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)

    logger.debug(f"Log level set to: {logging.getLevelName(logger.getEffectiveLevel())}")
