# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from environs import Env


class QueueConfig:
    max_jobs_per_dataset: int
    max_load_pct: int
    max_memory_pct: int
    mongo_database: str
    mongo_url: str
    sleep_seconds: int

    def __init__(self):
        env = Env(expand_vars=True)
        with env.prefixed("QUEUE_"):
            self.mongo_database = env.str(name="MONGO_DATABASE", default="datasets_server_queue")
            self.mongo_url = env.str(name="MONGO_URL", default="mongodb://localhost:27017")
            self.max_jobs_per_dataset = env.int(name="MAX_JOBS_PER_DATASET", default=1)
            self.max_load_pct = env.int(name="MAX_LOAD_PCT", default=70)
            self.max_memory_pct = env.int(name="MAX_MEMORY_PCT", default=80)
            self.sleep_seconds = env.int(name="SLEEP_SECONDS", default=15)
