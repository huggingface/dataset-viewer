# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import logging
from typing import Optional

from huggingface_hub._commit_api import CommitOperationDelete
from huggingface_hub.hf_api import HfApi
from libcommon.simple_cache import get_all_datasets


def delete_indexes(
    hf_endpoint: str,
    target_revision: str,
    hf_token: Optional[str] = None,
    committer_hf_token: Optional[str] = None,
) -> None:
    logging.info("delete old duckdb index files from refs/convert/parquet")
    dataset_names = get_all_datasets()
    hf_api = HfApi(endpoint=hf_endpoint, token=hf_token)
    committer_hf_api = HfApi(endpoint=hf_endpoint, token=committer_hf_token)
    num_total_datasets = len(dataset_names)
    num_analyzed_datasets = 0
    num_untouched_datasets = 0
    num_error_datasets = 0
    success_datasets = 0
    log_batch = 100

    def get_log() -> str:
        return (
            f"{num_analyzed_datasets} analyzed datasets (total: {num_total_datasets} datasets): "
            f"{num_untouched_datasets} already ok ({100 * num_untouched_datasets / num_analyzed_datasets:.2f}%), "
            f"{num_error_datasets} raised an exception ({100 * num_error_datasets / num_analyzed_datasets:.2f}%). "
        )

    for dataset in dataset_names:
        target_dataset_info = hf_api.dataset_info(repo_id=dataset, revision=target_revision, files_metadata=False)
        all_repo_files: set[str] = {f.rfilename for f in target_dataset_info.siblings}
        files_to_delete = [file for file in all_repo_files if file.endswith(".duckdb")]
        num_analyzed_datasets += 1
        if not files_to_delete:
            num_untouched_datasets += 1
            continue
        delete_operations = [CommitOperationDelete(path_in_repo=file) for file in files_to_delete]
        try:
            logging.info(f"deleting duckdb index files for {dataset=} {files_to_delete}")
            committer_hf_api.create_commit(
                repo_id=dataset,
                repo_type="dataset",
                revision=target_revision,
                operations=delete_operations,
                commit_message="Delete old duckdb index files",
                parent_commit=target_dataset_info.sha,
            )
        except Exception as e:
            logging.error(e)
            num_error_datasets += 1
        success_datasets += 1
        if num_analyzed_datasets % log_batch == 0:
            logging.info(get_log())
    logging.info(get_log())
    logging.info(f"old duckdb index files have been deleted from {target_revision}.")
