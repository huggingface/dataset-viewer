# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import List

from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority, Queue
from libcommon.simple_cache import get_outdated_split_full_names_for_step


def refresh_cache(processing_steps: List[ProcessingStep]) -> None:
    queue = Queue()
    for processing_step in processing_steps:
        current_version = processing_step.job_runner_version
        cache_records = get_outdated_split_full_names_for_step(processing_step.cache_kind, current_version)
        logging.info(
            (
                f"processing_step={processing_step.cache_kind} ",
                (
                    f"{len(cache_records)} cache entries are outdated for"
                    f" processing_step={processing_step.cache_kind} (version {current_version}). Creating jobs to"
                    " update them."
                ),
            )
        )

        for cache_info in cache_records:
            logging.debug(
                (
                    f"upsert_job for processing_step={processing_step.job_type}",
                    f"  dataset={cache_info.dataset} config={cache_info.config} split={cache_info.split}",
                ),
            )
            queue.upsert_job(
                job_type=processing_step.job_type,
                dataset=cache_info["dataset"],  # type: ignore
                config=cache_info["config"],
                split=cache_info["split"],
                force=False,
                priority=Priority.LOW,
            )
