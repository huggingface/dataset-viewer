# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging


def init_logging(level: int = logging.INFO) -> None:
    # force=True: heavy imports (datasets, torch, ...) can attach a handler to the root logger before
    # this runs, which would make basicConfig a no-op and leave the root logger at its default WARNING
    # level, silently dropping every root INFO/DEBUG log. force removes those handlers and applies ours.
    logging.basicConfig(level=level, format="%(levelname)s: %(asctime)s - %(name)s - %(message).5000s", force=True)
    # At DEBUG level, botocore logs every signed request with its headers, including the AWS session token.
    for logger_name in ("botocore", "aiobotocore", "s3fs"):
        logging.getLogger(logger_name).setLevel(max(level, logging.WARNING))
    logging.debug(f"Log level set to: {logging.getLevelName(logging.getLogger().getEffectiveLevel())}")
