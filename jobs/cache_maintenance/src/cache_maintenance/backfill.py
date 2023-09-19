# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libcommon.dataset import get_supported_dataset_infos
from libcommon.orchestrator import DatasetOrchestrator
from libcommon.processing_graph import ProcessingGraph
from libcommon.simple_cache import delete_dataset_responses, get_all_datasets
from libcommon.storage import StrPath
from libcommon.utils import Priority
from libcommon.viewer_utils.asset import delete_asset_dir


def backfill_cache(
    processing_graph: ProcessingGraph,
    hf_endpoint: str,
    cache_max_days: int,
    assets_directory: StrPath,
    cached_assets_directory: StrPath,
    hf_token: Optional[str] = None,
    error_codes_to_retry: Optional[list[str]] = None,
) -> None:
    logging.info("backfill supported datasets")
    supported_dataset_infos = get_supported_dataset_infos(hf_endpoint=hf_endpoint, hf_token=hf_token)
    logging.info(f"analyzing {len(supported_dataset_infos)} supported datasets")
    analyzed_datasets = 0
    backfilled_datasets = 0
    deleted_cache_records = 0
    total_created_jobs = 0
    log_batch = 100

    def get_log() -> str:
        return (
            f"{analyzed_datasets} analyzed datasets (total: {len(supported_dataset_infos)} datasets):"
            f" {backfilled_datasets} backfilled datasets ({100 * backfilled_datasets / analyzed_datasets:.2f}%), with"
            f" {total_created_jobs} created jobs."
        )

    supported_dataset_names = {dataset_info.id for dataset_info in supported_dataset_infos}
    existing_datasets = get_all_datasets("dataset-config-names")
    datasets_to_delete = existing_datasets.difference(supported_dataset_names)
    for dataset in datasets_to_delete:
        delete_asset_dir(dataset=dataset, directory=assets_directory)
        delete_asset_dir(dataset=dataset, directory=cached_assets_directory)
        datasets_cache_records = delete_dataset_responses(dataset=dataset)
        deleted_cache_records += datasets_cache_records if datasets_cache_records is not None else 0
        logging.debug(f"{datasets_cache_records} cache records deleted for {dataset=}")

    for dataset_info in supported_dataset_infos:
        analyzed_datasets += 1

        dataset = dataset_info.id
        if not dataset:
            logging.warning(f"dataset id not found for {dataset_info}")
            # should not occur
            continue
        if dataset_info.sha is None:
            logging.warning(f"dataset revision not found for {dataset_info}")
            # should not occur
            continue
        try:
            dataset_orchestrator = DatasetOrchestrator(dataset=dataset, processing_graph=processing_graph)
        except Exception as e:
            logging.warning(f"failed to create DatasetOrchestrator for {dataset_info}: {e}")
            continue
        try:
            created_jobs = dataset_orchestrator.backfill(
                revision=str(dataset_info.sha),
                priority=Priority.LOW,
                error_codes_to_retry=error_codes_to_retry,
                cache_max_days=cache_max_days,
            )
            if created_jobs > 0:
                backfilled_datasets += 1
            total_created_jobs += created_jobs
        except Exception as e:
            logging.warning(f"failed to backfill {dataset_info}: {e}")
            continue

        logging.debug(get_log())
        if analyzed_datasets % log_batch == 0:
            logging.info(get_log())

    logging.info(get_log())
    logging.info(f"{len(datasets_to_delete)} datasets were deleted with {deleted_cache_records} cached records")
    logging.info("backfill completed")
