# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libcommon.dataset import get_supported_dataset_infos
from libcommon.processing_graph import ProcessingGraph
from libcommon.state import DatasetState


def backfill_cache(
    processing_graph: ProcessingGraph,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> None:
    logging.info("backfill supported datasets")
    supported_dataset_infos = get_supported_dataset_infos(hf_endpoint=hf_endpoint, hf_token=hf_token)
    logging.info(f"analyzing {len(supported_dataset_infos)} supported datasets")
    analyzed_datasets = 0
    backfilled_datasets = 0
    total_created_jobs = 0
    log_batch = 100

    def get_log() -> str:
        return (
            f"{analyzed_datasets} analyzed datasets: {backfilled_datasets} backfilled datasets"
            f" ({100 * backfilled_datasets / analyzed_datasets:.2f}%), with {total_created_jobs} created jobs."
        )

    for dataset_info in supported_dataset_infos:
        analyzed_datasets += 1

        dataset = dataset_info.id
        if not dataset:
            # should not occur
            continue
        dataset_state = DatasetState(dataset=dataset, processing_graph=processing_graph, revision=dataset_info.sha)
        created_jobs = dataset_state.backfill()
        if created_jobs > 0:
            backfilled_datasets += 1
        total_created_jobs += created_jobs

        logging.debug(get_log())
        if analyzed_datasets % log_batch == 0:
            logging.info(get_log())

    logging.info(get_log())
    logging.info("backfill completed")
