# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from typing import Optional

from libcommon.dataset import get_supported_dataset_infos
from libcommon.simple_cache import delete_dataset_responses, get_all_datasets
from libcommon.storage import StrPath
from libcommon.viewer_utils.asset import delete_asset_dir

MINIMUM_SUPPORTED_DATASETS = 20000


def clean_cache(
    hf_endpoint: str,
    assets_directory: StrPath,
    cached_assets_directory: StrPath,
    hf_token: Optional[str] = None,
) -> None:
    """
    Delete cache records and assets from datasets that no longer exist on the hub
    """
    logging.info("clean unsupported datasets")
    supported_dataset_infos = get_supported_dataset_infos(hf_endpoint=hf_endpoint, hf_token=hf_token)
    if len(supported_dataset_infos) < MINIMUM_SUPPORTED_DATASETS:
        logging.warning(f"only {len(supported_dataset_infos)} datasets were found, the clean up will be cancelled")
        return
    deleted_datasets = 0
    deleted_cache_records = 0
    log_batch = 100
    supported_dataset_names = {dataset_info.id for dataset_info in supported_dataset_infos}
    existing_datasets = get_all_datasets("dataset-config-names")
    datasets_to_delete = existing_datasets.difference(supported_dataset_names)

    def get_log() -> str:
        return (
            f"{deleted_datasets} deleted datasets (total: {len(datasets_to_delete)} datasets):"
            f"with {deleted_cache_records} cache records."
        )

    for dataset in datasets_to_delete:
        deleted_datasets += 1

        # delete assets
        delete_asset_dir(dataset=dataset, directory=assets_directory)
        delete_asset_dir(dataset=dataset, directory=cached_assets_directory)

        # delete all cache records
        datasets_cache_records = delete_dataset_responses(dataset=dataset)
        deleted_cache_records += datasets_cache_records if datasets_cache_records is not None else 0

        logging.debug(f"{datasets_cache_records} cache records deleted for {dataset=}")
        logging.debug(get_log())
        if deleted_datasets % log_batch == 0:
            logging.info(get_log())

    logging.info(get_log())
    logging.info("clean completed")
