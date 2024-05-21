# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import logging
from http import HTTPStatus

from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import (
    CachedArtifactNotFoundError,
    get_previous_step_or_raise,
    get_response,
)

from worker.dtos import JobResult, PresidioEntitiesCountResponse
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner


def compute_presidio_entities_count_response(dataset: str) -> tuple[PresidioEntitiesCountResponse, float]:
    logging.info(f"compute 'dataset-presidio-entities-count' for {dataset=}")

    split_names_response = get_previous_step_or_raise(kind="dataset-split-names", dataset=dataset)
    content = split_names_response["content"]
    if "splits" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'splits'.")

    scanned_columns = set()
    presidio_entities_count_response = PresidioEntitiesCountResponse(
        {
            "scanned_columns": [],
            "num_rows_with_person_entities": 0,
            "num_rows_with_phone_number_entities": 0,
            "num_rows_with_email_address_entities": 0,
            "num_rows_with_sensitive_pii": 0,
            "num_scanned_rows": 0,
            "has_scanned_columns": False,
            "full_scan": True,
        }
    )
    try:
        total = 0
        pending = 0
        for split_item in content["splits"]:
            config = split_item["config"]
            split = split_item["split"]
            total += 1
            try:
                response = get_response(kind="split-presidio-scan", dataset=dataset, config=config, split=split)
            except CachedArtifactNotFoundError:
                logging.debug("No response found in previous step for this dataset: 'split-presidio-scan'.")
                pending += 1
                continue
            if response["http_status"] != HTTPStatus.OK:
                logging.debug(f"Previous step gave an error: {response['http_status']}.")
                continue
            split_presidio_scan_content = response["content"]
            scanned_columns.update(split_presidio_scan_content["scanned_columns"])
            if not split_presidio_scan_content["full_scan"]:
                presidio_entities_count_response["full_scan"] = False
            presidio_entities_count_response["num_rows_with_person_entities"] += split_presidio_scan_content[
                "num_rows_with_person_entities"
            ]
            presidio_entities_count_response["num_rows_with_phone_number_entities"] += split_presidio_scan_content[
                "num_rows_with_phone_number_entities"
            ]
            presidio_entities_count_response["num_rows_with_email_address_entities"] += split_presidio_scan_content[
                "num_rows_with_email_address_entities"
            ]
            presidio_entities_count_response["num_rows_with_sensitive_pii"] += split_presidio_scan_content[
                "num_rows_with_credit_card_entities"
            ]
            presidio_entities_count_response["num_rows_with_sensitive_pii"] += split_presidio_scan_content[
                "num_rows_with_us_ssn_entities"
            ]
            presidio_entities_count_response["num_rows_with_sensitive_pii"] += split_presidio_scan_content[
                "num_rows_with_us_passport_entities"
            ]
            presidio_entities_count_response["num_rows_with_sensitive_pii"] += split_presidio_scan_content[
                "num_rows_with_iban_code_entities"
            ]
            presidio_entities_count_response["num_scanned_rows"] += split_presidio_scan_content["num_scanned_rows"]
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    presidio_entities_count_response["scanned_columns"] = sorted(scanned_columns)
    presidio_entities_count_response["has_scanned_columns"] = (
        len(presidio_entities_count_response["scanned_columns"]) > 0
    )
    progress = (total - pending) / total if total else 1.0

    return (presidio_entities_count_response, progress)


class DatasetPresidioEntitiesCountJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-presidio-entities-count"

    def compute(self) -> JobResult:
        response_content, progress = compute_presidio_entities_count_response(dataset=self.dataset)
        return JobResult(response_content, progress=progress)
