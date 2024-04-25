# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from collections import Counter
from itertools import islice, count
from pathlib import Path
from typing import Iterable, Optional, TypeVar

from datasets import get_dataset_config_info
from libcommon.dtos import JobInfo, Row
from libcommon.exceptions import (
    DatasetWithScriptNotSupportedError,
    InfoError,
    TooManyColumnsError,
)
from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine

from worker.config import AppConfig, PresidioEntitiesScanConfig
from worker.dtos import CompleteJobResult, PresidioEntitiesScanResponse, PresidioEntity
from worker.job_runners.split.split_job_runner import SplitJobRunnerWithDatasetsCache
from worker.utils import get_rows_or_raise, resolve_trust_remote_code


T = TypeVar("T")


def batched(it: Iterable[T], n: int) -> Iterable[list[T]]:
    it = iter(it)
    while batch := list(islice(it, n)):
        yield batch


def analyze(batch_analyzer: BatchAnalyzerEngine, batch: list[dict[str, str]], indices: Iterable[int], column_names: list[str]) -> list[PresidioEntity]:
    texts = [f"The following is {column_name} data:\n\n{example[column_name] or ''}" for example in batch for column_name in column_names]
    return [
        PresidioEntity(
            text=texts[i][recognizer_result.start:recognizer_result.end],
            type=recognizer_result.entity_type,
            row_idx=row_idx,
            column_name=column_name
        )
        for i, row_idx, recognizer_results in zip(count(), indices, batch_analyzer.analyze_iterator(texts, language="en", score_threshold=0.8))
        for column_name, recognizer_result in zip(column_names, recognizer_results)
        if recognizer_result.start >= len(f"The following is {column_name} data:\n\n")
    ]


def presidio_scan_entities(rows: list[Row], scanned_columns: list[str]) -> list[PresidioEntity]:
    pass # TODO(QL): Implement


def compute_presidio_entities_scan_response(
    dataset: str,
    config: str,
    split: str,
    hf_token: Optional[str],
    rows_max_number: int,
    columns_max_number: int,
    urls_number_per_batch: int,
    spawning_token: Optional[str],
    max_concurrent_requests_number: int,
    max_requests_per_second: int,
    spawning_url: str,
    dataset_scripts_allow_list: list[str],
) -> PresidioEntitiesScanResponse:
    """
    Get the response of 'split-presidio-scan' cache for a specific split of a dataset from huggingface.co.
    The response is not used directly in the API but it is an input for 'config-presidio-scan' processing step.
    Note that only image URLs are scanned, see image_url_columns.py for details about the detection heuristic.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.
        config (`str`):
            A configuration name.
        split (`str`):
            A split name.
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
        rows_max_number (`int`):
            The maximum number of rows of the response.
        columns_max_number (`int`):
            The maximum number of supported columns.
        urls_number_per_batch (`int`):
            The number of batch URLs to be sent to spawning service.
        spawning_token (`str`, *optional*):
            An authentication token to use spawning service (See https://api.spawning.ai/spawning-api)
        max_concurrent_requests_number (`int`):
            The maximum number of requests to be processed concurrently.
        max_requests_per_second (`int`):
            The maximum number of requests to be processed by second.
        spawning_url (`str`):
            Spawgning API URL
        dataset_scripts_allow_list (`list[str]`):
            List of datasets for which we support dataset scripts.
            Unix shell-style wildcards also work in the dataset name for namespaced datasets,
            for example `some_namespace/*` to refer to all the datasets in the `some_namespace` namespace.
            The keyword `{{ALL_DATASETS_WITH_NO_NAMESPACE}}` refers to all the datasets without namespace.

    Raises:
        [~`libcommon.simple_cache.CachedArtifactError`]:
          If the previous step gave an error.
        [~`libcommon.exceptions.PreviousStepFormatError`]:
          If the content of the previous step has not the expected format
        [~`libcommon.exceptions.InfoError`]:
          If the config info could not be obtained using the datasets library.
        [~`libcommon.exceptions.TooManyColumnsError`]:
          If the number of columns (features) exceeds the maximum supported number of columns.
        [~`libcommon.exceptions.StreamingRowsError`]:
          If the split rows could not be obtained using the datasets library in streaming mode.
        [~`libcommon.exceptions.NormalRowsError`]:
          If the split rows could not be obtained using the datasets library in normal mode.
        [~`libcommon.exceptions.DatasetWithScriptNotSupportedError`]:
            If the dataset has a dataset script and is not in the allow list.

    Returns:
        `PresidioEntitiesScanResponse`: An object with the lists of opt-in/opt-out urls
    """
    logging.info(f"compute 'split-presidio-scan' for {dataset=} {config=} {split=}")
    trust_remote_code = resolve_trust_remote_code(dataset=dataset, allow_list=dataset_scripts_allow_list)

    # get the info
    try:
        info = get_dataset_config_info(
            path=dataset, config_name=config, token=hf_token, trust_remote_code=trust_remote_code
        )
    except Exception as err:
        if isinstance(err, ValueError) and "trust_remote_code" in str(err):
            raise DatasetWithScriptNotSupportedError(
                "The dataset viewer doesn't support this dataset because it runs "
                "arbitrary python code. Please open a discussion in the discussion tab "
                "if you think this is an error and tag @lhoestq and @severo."
            ) from err
        raise InfoError(
            f"The info cannot be fetched for the config '{config}' of the dataset.",
            cause=err,
        ) from err

    scanned_columns = ...
    if not scanned_columns:
        return PresidioEntitiesScanResponse(
            scanned_columns=scanned_columns,
            num_in_vehicle_registration_entities=0,
            num_organization_entities=0,
            num_sg_nric_fin_entities=0,
            num_person_entities=0,
            num_credit_card_entities=0,
            num_medical_license_entities=0,
            num_nrp_entities=0,
            num_us_ssn_entities=0,
            num_crypto_entities=0,
            num_date_time_entities=0,
            num_location_entities=0,
            num_us_driver_license_entities=0,
            num_phone_number_entities=0,
            num_url_entities=0,
            num_us_passport_entities=0,
            num_age_entities=0,
            num_au_acn_entities=0,
            num_email_address_entities=0,
            num_in_pan_entities=0,
            num_ip_address_entities=0,
            num_id_entities=0,
            num_us_bank_number_entities=0,
            num_in_aadhaar_entities=0,
            num_us_itin_entities=0,
            num_au_medicare_entities=0,
            num_iban_code_entities=0,
            num_au_tfn_entities=0,
            num_uk_nhs_entities=0,
            num_email_entities=0,
            num_au_abn_entities=0,
            num_scanned_rows=0,
            has_scanned_columns=False,
            full_scan=None,
            entities=[],
        )

    if len(scanned_columns) > columns_max_number:
        raise TooManyColumnsError(
            f"The number of columns ({len(scanned_columns)}) exceeds the maximum supported number of columns to scan"
            f" ({columns_max_number})."
        )

    # get the rows
    rows_content = get_rows_or_raise(
        dataset=dataset,
        config=config,
        split=split,
        info=info,
        rows_max_number=rows_max_number,
        token=hf_token,
        column_names=scanned_columns,
        trust_remote_code=trust_remote_code,
    )
    rows = rows_content.rows

    # scan the texts for presidio entities
    num_scanned_rows = len(rows)
    presidio_entities = presidio_scan_entities(rows, scanned_columns=scanned_columns)
    counter = Counter(presidio_entity.type for presidio_entity in presidio_entities)

    # return scan result
    return PresidioEntitiesScanResponse(
        scanned_columns=scanned_columns,
        num_in_vehicle_registration_entities=counter.get("IN_VEHICLE_REGISTRATION", 0),
        num_organization_entities=counter.get("ORGANIZATION", 0),
        num_sg_nric_fin_entities=counter.get("SG_NRIC_FIN", 0),
        num_person_entities=counter.get("PERSON", 0),
        num_credit_card_entities=counter.get("CREDIT_CARD", 0),
        num_medical_license_entities=counter.get("MEDICAL_LICENSE", 0),
        num_nrp_entities=counter.get("NRP", 0),
        num_us_ssn_entities=counter.get("US_SSN", 0),
        num_crypto_entities=counter.get("CRYPTO", 0),
        num_date_time_entities=counter.get("DATE_TIME", 0),
        num_location_entities=counter.get("LOCATION", 0),
        num_us_driver_license_entities=counter.get("US_DRIVER_LICENSE", 0),
        num_phone_number_entities=counter.get("PHONE_NUMBER", 0),
        num_url_entities=counter.get("URL", 0),
        num_us_passport_entities=counter.get("US_PASSPORT", 0),
        num_age_entities=counter.get("AGE", 0),
        num_au_acn_entities=counter.get("AU_ACN", 0),
        num_email_address_entities=counter.get("EMAIL_ADDRESS", 0),
        num_in_pan_entities=counter.get("IN_PAN", 0),
        num_ip_address_entities=counter.get("IP_ADDRESS", 0),
        num_id_entities=counter.get("ID", 0),
        num_us_bank_number_entities=counter.get("US_BANK_NUMBER", 0),
        num_in_aadhaar_entities=counter.get("IN_AADHAAR", 0),
        num_us_itin_entities=counter.get("US_ITIN", 0),
        num_au_medicare_entities=counter.get("AU_MEDICARE", 0),
        num_iban_code_entities=counter.get("IBAN_CODE", 0),
        num_au_tfn_entities=counter.get("AU_TFN", 0),
        num_uk_nhs_entities=counter.get("UK_NHS", 0),
        num_email_entities=counter.get("EMAIL", 0),
        num_au_abn_entities=counter.get("AU_ABN", 0),
        num_scanned_rows=num_scanned_rows,
        has_scanned_columns=True,
        full_scan=rows_content.all_fetched,
        entities=presidio_entities,
    )


class SplitPresidioEntitiesScanJobRunner(SplitJobRunnerWithDatasetsCache):
    presidio_entities_scan_config: PresidioEntitiesScanConfig

    @staticmethod
    def get_job_type() -> str:
        return "split-presidio-scan"

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        hf_datasets_cache: Path,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            hf_datasets_cache=hf_datasets_cache,
        )
        self.presidio_entities_scan_config = app_config.presidio_scan

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_presidio_entities_scan_response(
                dataset=self.dataset,
                config=self.config,
                split=self.split,
                hf_token=self.app_config.common.hf_token,
                rows_max_number=self.presidio_entities_scan_config.rows_max_number,
                columns_max_number=self.presidio_entities_scan_config.columns_max_number,
                dataset_scripts_allow_list=self.app_config.common.dataset_scripts_allow_list,
            )
        )
