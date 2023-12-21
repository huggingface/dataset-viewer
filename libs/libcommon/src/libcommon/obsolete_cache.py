# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from typing import Optional, TypedDict

from libcommon.simple_cache import (
    delete_dataset_responses,
    get_all_datasets,
    get_cache_count_for_dataset,
)
from libcommon.storage_client import StorageClient

MINIMUM_SUPPORTED_DATASETS = 20_000


class NotEnoughSupportedDatasetsError(Exception):
    pass


class DatasetCacheReport(TypedDict):
    dataset: str
    cache_records: Optional[int]


def get_supported_dataset_names(
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> set[str]:
    # supported_dataset_infos = get_supported_dataset_infos(hf_endpoint=hf_endpoint, hf_token=hf_token)
    # return {dataset_info.id for dataset_info in supported_dataset_infos}
    return {"glue"} if hf_token is None and hf_endpoint == "aaa" else {"glue"}
    # TODO: restore


def get_obsolete_cache(
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> list[DatasetCacheReport]:
    supported_dataset_names = get_supported_dataset_names(hf_endpoint=hf_endpoint, hf_token=hf_token)
    existing_datasets = get_all_datasets()
    datasets_to_delete = existing_datasets.difference(supported_dataset_names)
    return [
        DatasetCacheReport(dataset=dataset, cache_records=get_cache_count_for_dataset(dataset=dataset))
        for dataset in datasets_to_delete
    ]


def delete_obsolete_cache(
    hf_endpoint: str,
    cached_assets_storage_client: StorageClient,
    assets_storage_client: StorageClient,
    hf_token: Optional[str] = None,
) -> list[DatasetCacheReport]:
    logging.info("deleting obsolete cache")
    supported_dataset_names = get_supported_dataset_names(hf_endpoint=hf_endpoint, hf_token=hf_token)
    if len(supported_dataset_names) < MINIMUM_SUPPORTED_DATASETS:
        raise NotEnoughSupportedDatasetsError(f"only {len(supported_dataset_names)} datasets were found")

    existing_datasets = get_all_datasets()
    datasets_to_delete = existing_datasets.difference(supported_dataset_names)

    deletion_report = []
    deleted_cached_records = 0
    for dataset in datasets_to_delete:
        # delete cache records
        datasets_cache_records = delete_dataset_responses(dataset=dataset)
        deleted_cached_records += datasets_cache_records if datasets_cache_records is not None else 0
        if datasets_cache_records is not None and datasets_cache_records > 0:
            # delete assets
            cached_assets_storage_client.delete_dataset_directory(dataset)
            assets_storage_client.delete_dataset_directory(dataset)
            logging.info(f"{dataset} has been deleted with {datasets_cache_records} cache records")
        else:
            logging.debug(f"unable to delete {dataset}")
        deletion_report.append(DatasetCacheReport(dataset=dataset, cache_records=datasets_cache_records))
    logging.info(f"{len(deletion_report)} datasets have been removed with {deleted_cached_records} cache records.")

    return deletion_report
