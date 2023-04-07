# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

import libcommon
from libcommon.operations import update_dataset
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority


def backfill_cache(
    init_processing_steps: list[ProcessingStep],
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> None:
    logging.info("backfill init processing steps for supported datasets")
    supported_datasets = libcommon.dataset.get_supported_datasets(hf_endpoint=hf_endpoint, hf_token=hf_token)
    logging.info(f"about to backfill {len(supported_datasets)} datasets")
    backfilled_datasets = 0
    log_batch = 100
    for dataset in libcommon.dataset.get_supported_datasets(hf_endpoint=hf_endpoint, hf_token=hf_token):
        update_dataset(
            dataset=dataset,
            init_processing_steps=init_processing_steps,
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            force=False,
            priority=Priority.LOW,
            do_check_support=False,
        )
        backfilled_datasets += 1
        if backfilled_datasets % log_batch == 0:
            logging.info(f"{backfilled_datasets} datasets have been backfilled")
    logging.info("backfill completed")
