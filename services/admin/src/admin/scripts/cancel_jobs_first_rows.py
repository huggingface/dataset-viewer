# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libqueue.queue import Queue

from admin.config import AppConfig
from admin.utils import JobType

if __name__ == "__main__":
    app_config = AppConfig()
    Queue(type=JobType.FIRST_ROWS.value).cancel_started_jobs()
    logging.info("all the started jobs in the first_rows/ queue have been cancelled and re-enqueued")
