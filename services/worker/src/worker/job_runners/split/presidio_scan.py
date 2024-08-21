# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
import re
from collections import Counter
from collections.abc import Iterable
from itertools import count
from pathlib import Path
from typing import Any, Optional

from datasets import DatasetInfo, Features, Value
from datasets.features.features import FeatureType, _visit
from libcommon.dtos import JobInfo, Row
from libcommon.exceptions import (
    PresidioScanNotEnabledForThisDataset,
    PreviousStepFormatError,
    TooManyColumnsError,
)
from libcommon.simple_cache import get_previous_step_or_raise
from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine, RecognizerResult

from worker.config import AppConfig, PresidioEntitiesScanConfig
from worker.dtos import CompleteJobResult, ConfigParquetAndInfoResponse, PresidioEntitiesScanResponse, PresidioEntity
from worker.job_runners.split.split_job_runner import SplitJobRunnerWithDatasetsCache
from worker.utils import batched, get_rows_or_raise, resolve_trust_remote_code

BATCH_SIZE = 10
batch_analyzer: Optional[BatchAnalyzerEngine] = None


def mask(text: str) -> str:
    return " ".join(
        word[: min(2, len(word) - 1)] + re.sub("[A-Za-z0-9]", "*", word[min(2, len(word) - 1) :])
        for word in text.split(" ")
    )


def get_strings(row_content: Any) -> str:
    if isinstance(row_content, str):
        return row_content
    if isinstance(row_content, dict):
        row_content = list(row_content.values())
    if isinstance(row_content, list):
        str_items = (get_strings(row_content_item) for row_content_item in row_content)
        return "\n".join(str_item for str_item in str_items if str_item)
    return ""


def _simple_analyze_iterator_cache(
    batch_analyzer: BatchAnalyzerEngine,
    texts: Iterable[str],
    language: str,
    score_threshold: float,
    cache: dict[str, list[RecognizerResult]],
) -> list[list[RecognizerResult]]:
    not_cached_results = iter(
        batch_analyzer.analyze_iterator(
            (text for text in texts if text not in cache), language=language, score_threshold=score_threshold
        )
    )
    results = [cache[text] if text in cache else next(not_cached_results) for text in texts]
    # cache the last results
    cache.clear()
    cache.update(dict(zip(texts, results)))
    return results


def analyze(
    batch_analyzer: BatchAnalyzerEngine,
    batch: list[dict[str, str]],
    indices: Iterable[int],
    scanned_columns: list[str],
    columns_descriptions: list[str],
    cache: Optional[dict[str, list[RecognizerResult]]] = None,
) -> list[PresidioEntity]:
    cache = {} if cache is None else cache
    texts = [
        f"The following is {columns_description} data:\n\n{example[column_name] or ''}"
        for example in batch
        for column_name, columns_description in zip(scanned_columns, columns_descriptions)
    ]
    return [
        PresidioEntity(
            text=texts[i * len(scanned_columns) + j][recognizer_result.start : recognizer_result.end],
            type=recognizer_result.entity_type,
            row_idx=row_idx,
            column_name=column_name,
        )
        for i, row_idx, recognizer_row_results in zip(
            count(),
            indices,
            batched(
                _simple_analyze_iterator_cache(batch_analyzer, texts, language="en", score_threshold=0.8, cache=cache),
                len(scanned_columns),
            ),
        )
        for j, column_name, columns_description, recognizer_results in zip(
            count(), scanned_columns, columns_descriptions, recognizer_row_results
        )
        for recognizer_result in recognizer_results
        if recognizer_result.start >= len(f"The following is {columns_description} data:\n\n")
    ]


def presidio_scan_entities(
    rows: list[Row],
    scanned_columns: list[str],
    columns_descriptions: list[str],
    max_text_length: int,
    disable_masks: bool = False,
) -> list[PresidioEntity]:
    global batch_analyzer
    cache: dict[str, list[RecognizerResult]] = {}
    if batch_analyzer is None:
        batch_analyser = BatchAnalyzerEngine(AnalyzerEngine())
    presidio_entities: list[PresidioEntity] = []
    rows_with_scanned_columns_only = (
        {column_name: get_strings(row[column_name])[:max_text_length] for column_name in scanned_columns}
        for row in rows
    )
    for indices, batch in batched(rows_with_scanned_columns_only, BATCH_SIZE, with_indices=True):
        for presidio_entitiy in analyze(
            batch_analyzer=batch_analyser,
            batch=batch,
            indices=indices,
            scanned_columns=scanned_columns,
            columns_descriptions=columns_descriptions,
            cache=cache,
        ):
            presidio_entities.append(
                PresidioEntity(
                    text=presidio_entitiy["text"] if disable_masks else mask(presidio_entitiy["text"]),
                    type=presidio_entitiy["type"],
                    row_idx=presidio_entitiy["row_idx"],
                    column_name=presidio_entitiy["column_name"],
                )
            )
    return presidio_entities


def get_columns_with_strings(features: Features) -> list[str]:
    columns_with_strings: list[str] = []

    for column, feature in features.items():
        str_column = str(column)
        with_string = False

        def classify(feature: FeatureType) -> None:
            nonlocal with_string
            if isinstance(feature, Value) and feature.dtype == "string":
                with_string = True

        _visit(feature, classify)
        if with_string:
            columns_with_strings.append(str_column)
    return columns_with_strings


def get_column_description(column_name: str, feature: FeatureType) -> str:
    nested_fields: list[str] = []

    def get_nested_field_names(feature: FeatureType) -> None:
        nonlocal nested_fields
        if isinstance(feature, dict):
            nested_fields += list(feature)

    _visit(feature, get_nested_field_names)
    return f"{column_name} (with {', '.join(nested_fields)})" if nested_fields else column_name


def compute_presidio_entities_scan_response(
    dataset: str,
    config: str,
    split: str,
    hf_token: Optional[str],
    rows_max_number: int,
    columns_max_number: int,
    max_text_length: int,
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
        max_text_length (`int`):
            The maximum text length considered by the scanner.
        dataset_scripts_allow_list (`list[str]`):
            List of datasets for which we support dataset scripts.
            Unix shell-style wildcards also work in the dataset name for namespaced datasets,
            for example `some_namespace/*` to refer to all the datasets in the `some_namespace` namespace.

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
    if not (
        "email" in dataset
        or "pii" in dataset
        or "presidio" in dataset
        or "ssn" in dataset
        or "DVUser/" in dataset
        or dataset in enabled_datasets
    ):
        raise PresidioScanNotEnabledForThisDataset(dataset)
    logging.info(f"compute 'split-presidio-scan' for {dataset=} {config=} {split=}")
    trust_remote_code = resolve_trust_remote_code(dataset=dataset, allow_list=dataset_scripts_allow_list)

    # get the first rows from previous job
    parquet_and_info_response = get_previous_step_or_raise(
        kind="config-parquet-and-info",
        dataset=dataset,
        config=config,
    )
    try:
        upstream_response_content = ConfigParquetAndInfoResponse(
            parquet_files=parquet_and_info_response["content"]["parquet_files"],
            dataset_info=parquet_and_info_response["content"]["dataset_info"],
            partial=parquet_and_info_response["content"]["partial"],
            estimated_dataset_info=parquet_and_info_response["content"].get("estimated_dataset_info"),
        )
        info = DatasetInfo.from_dict(upstream_response_content["dataset_info"])
        if info.features is None:
            raise PreviousStepFormatError("Previous step did not return the expected content (missing features).")
        features = info.features
    except KeyError as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    scanned_columns = get_columns_with_strings(features)
    columns_descriptions = [
        get_column_description(column_name, features[column_name]) for column_name in scanned_columns
    ]
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
            num_rows_with_in_vehicle_registration_entities=0,
            num_rows_with_organization_entities=0,
            num_rows_with_sg_nric_fin_entities=0,
            num_rows_with_person_entities=0,
            num_rows_with_credit_card_entities=0,
            num_rows_with_medical_license_entities=0,
            num_rows_with_nrp_entities=0,
            num_rows_with_us_ssn_entities=0,
            num_rows_with_crypto_entities=0,
            num_rows_with_date_time_entities=0,
            num_rows_with_location_entities=0,
            num_rows_with_us_driver_license_entities=0,
            num_rows_with_phone_number_entities=0,
            num_rows_with_url_entities=0,
            num_rows_with_us_passport_entities=0,
            num_rows_with_age_entities=0,
            num_rows_with_au_acn_entities=0,
            num_rows_with_email_address_entities=0,
            num_rows_with_in_pan_entities=0,
            num_rows_with_ip_address_entities=0,
            num_rows_with_id_entities=0,
            num_rows_with_us_bank_number_entities=0,
            num_rows_with_in_aadhaar_entities=0,
            num_rows_with_us_itin_entities=0,
            num_rows_with_au_medicare_entities=0,
            num_rows_with_iban_code_entities=0,
            num_rows_with_au_tfn_entities=0,
            num_rows_with_uk_nhs_entities=0,
            num_rows_with_email_entities=0,
            num_rows_with_au_abn_entities=0,
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
    presidio_entities = presidio_scan_entities(
        rows,
        scanned_columns=scanned_columns,
        columns_descriptions=columns_descriptions,
        max_text_length=max_text_length,
    )
    entity_type_counter = Counter(presidio_entity["type"] for presidio_entity in presidio_entities)
    entity_type_and_row_idx_pairs = set(
        (presidio_entity["type"], presidio_entity["row_idx"]) for presidio_entity in presidio_entities
    )
    rows_per_entity_type_counter = Counter(entity_type for entity_type, _ in entity_type_and_row_idx_pairs)

    # return scan result
    return PresidioEntitiesScanResponse(
        scanned_columns=scanned_columns,
        num_in_vehicle_registration_entities=entity_type_counter.get("IN_VEHICLE_REGISTRATION", 0),
        num_organization_entities=entity_type_counter.get("ORGANIZATION", 0),
        num_sg_nric_fin_entities=entity_type_counter.get("SG_NRIC_FIN", 0),
        num_person_entities=entity_type_counter.get("PERSON", 0),
        num_credit_card_entities=entity_type_counter.get("CREDIT_CARD", 0),
        num_medical_license_entities=entity_type_counter.get("MEDICAL_LICENSE", 0),
        num_nrp_entities=entity_type_counter.get("NRP", 0),
        num_us_ssn_entities=entity_type_counter.get("US_SSN", 0),
        num_crypto_entities=entity_type_counter.get("CRYPTO", 0),
        num_date_time_entities=entity_type_counter.get("DATE_TIME", 0),
        num_location_entities=entity_type_counter.get("LOCATION", 0),
        num_us_driver_license_entities=entity_type_counter.get("US_DRIVER_LICENSE", 0),
        num_phone_number_entities=entity_type_counter.get("PHONE_NUMBER", 0),
        num_url_entities=entity_type_counter.get("URL", 0),
        num_us_passport_entities=entity_type_counter.get("US_PASSPORT", 0),
        num_age_entities=entity_type_counter.get("AGE", 0),
        num_au_acn_entities=entity_type_counter.get("AU_ACN", 0),
        num_email_address_entities=entity_type_counter.get("EMAIL_ADDRESS", 0),
        num_in_pan_entities=entity_type_counter.get("IN_PAN", 0),
        num_ip_address_entities=entity_type_counter.get("IP_ADDRESS", 0),
        num_id_entities=entity_type_counter.get("ID", 0),
        num_us_bank_number_entities=entity_type_counter.get("US_BANK_NUMBER", 0),
        num_in_aadhaar_entities=entity_type_counter.get("IN_AADHAAR", 0),
        num_us_itin_entities=entity_type_counter.get("US_ITIN", 0),
        num_au_medicare_entities=entity_type_counter.get("AU_MEDICARE", 0),
        num_iban_code_entities=entity_type_counter.get("IBAN_CODE", 0),
        num_au_tfn_entities=entity_type_counter.get("AU_TFN", 0),
        num_uk_nhs_entities=entity_type_counter.get("UK_NHS", 0),
        num_email_entities=entity_type_counter.get("EMAIL", 0),
        num_au_abn_entities=entity_type_counter.get("AU_ABN", 0),
        num_rows_with_in_vehicle_registration_entities=rows_per_entity_type_counter.get("IN_VEHICLE_REGISTRATION", 0),
        num_rows_with_organization_entities=rows_per_entity_type_counter.get("ORGANIZATION", 0),
        num_rows_with_sg_nric_fin_entities=rows_per_entity_type_counter.get("SG_NRIC_FIN", 0),
        num_rows_with_person_entities=rows_per_entity_type_counter.get("PERSON", 0),
        num_rows_with_credit_card_entities=rows_per_entity_type_counter.get("CREDIT_CARD", 0),
        num_rows_with_medical_license_entities=rows_per_entity_type_counter.get("MEDICAL_LICENSE", 0),
        num_rows_with_nrp_entities=rows_per_entity_type_counter.get("NRP", 0),
        num_rows_with_us_ssn_entities=rows_per_entity_type_counter.get("US_SSN", 0),
        num_rows_with_crypto_entities=rows_per_entity_type_counter.get("CRYPTO", 0),
        num_rows_with_date_time_entities=rows_per_entity_type_counter.get("DATE_TIME", 0),
        num_rows_with_location_entities=rows_per_entity_type_counter.get("LOCATION", 0),
        num_rows_with_us_driver_license_entities=rows_per_entity_type_counter.get("US_DRIVER_LICENSE", 0),
        num_rows_with_phone_number_entities=rows_per_entity_type_counter.get("PHONE_NUMBER", 0),
        num_rows_with_url_entities=rows_per_entity_type_counter.get("URL", 0),
        num_rows_with_us_passport_entities=rows_per_entity_type_counter.get("US_PASSPORT", 0),
        num_rows_with_age_entities=rows_per_entity_type_counter.get("AGE", 0),
        num_rows_with_au_acn_entities=rows_per_entity_type_counter.get("AU_ACN", 0),
        num_rows_with_email_address_entities=rows_per_entity_type_counter.get("EMAIL_ADDRESS", 0),
        num_rows_with_in_pan_entities=rows_per_entity_type_counter.get("IN_PAN", 0),
        num_rows_with_ip_address_entities=rows_per_entity_type_counter.get("IP_ADDRESS", 0),
        num_rows_with_id_entities=rows_per_entity_type_counter.get("ID", 0),
        num_rows_with_us_bank_number_entities=rows_per_entity_type_counter.get("US_BANK_NUMBER", 0),
        num_rows_with_in_aadhaar_entities=rows_per_entity_type_counter.get("IN_AADHAAR", 0),
        num_rows_with_us_itin_entities=rows_per_entity_type_counter.get("US_ITIN", 0),
        num_rows_with_au_medicare_entities=rows_per_entity_type_counter.get("AU_MEDICARE", 0),
        num_rows_with_iban_code_entities=rows_per_entity_type_counter.get("IBAN_CODE", 0),
        num_rows_with_au_tfn_entities=rows_per_entity_type_counter.get("AU_TFN", 0),
        num_rows_with_uk_nhs_entities=rows_per_entity_type_counter.get("UK_NHS", 0),
        num_rows_with_email_entities=rows_per_entity_type_counter.get("EMAIL", 0),
        num_rows_with_au_abn_entities=rows_per_entity_type_counter.get("AU_ABN", 0),
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
                max_text_length=self.presidio_entities_scan_config.max_text_length,
                dataset_scripts_allow_list=self.app_config.common.dataset_scripts_allow_list,
            )
        )


# fmt: off
top_2k_most_liked_datasets = {
    "fka/awesome-chatgpt-prompts", "Open-Orca/OpenOrca", "OpenAssistant/oasst1", "HuggingFaceFW/fineweb", "gsdf/EasyNegative", "Anthropic/hh-rlhf", "togethercomputer/RedPajama-Data-1T", "Nerfgun3/bad_prompt", "tiiuae/falcon-refinedweb", "allenai/dolma",
    "anon8231489123/ShareGPT_Vicuna_unfiltered", "bigcode/the-stack", "QingyiSi/Alpaca-CoT", "databricks/databricks-dolly-15k", "tatsu-lab/alpaca", "teknium/OpenHermes-2.5", "JosephusCheung/GuanacoDataset", "legacy-datasets/wikipedia", "HuggingFaceTB/cosmopedia", "m-a-p/COIG-CQIA",
    "lmsys/lmsys-chat-1m", "poloclub/diffusiondb", "liwu/MNBVC", "Gustavosta/Stable-Diffusion-Prompts", "BAAI/COIG", "uonlp/CulturaX", "yahma/alpaca-cleaned", "roneneldan/TinyStories", "stingning/ultrachat", "wikimedia/wikipedia",
    "GAIR/lima", "HuggingFaceH4/no_robots", "cognitivecomputations/dolphin", "cerebras/SlimPajama-627B", "timdettmers/openassistant-guanaco", "HuggingFaceH4/ultrachat_200k", "EleutherAI/pile", "liuhaotian/LLaVA-Instruct-150K", "b-mc2/sql-create-context", "garage-bAInd/Open-Platypus",
    "bigcode/starcoderdata", "microsoft/orca-math-word-problems-200k", "imagenet-1k", "nyu-mll/glue", "bigcode/the-stack-dedup", "togethercomputer/RedPajama-Data-V2", "gretelai/synthetic_text_to_sql", "allenai/objaverse", "Skylion007/openwebtext", "Salesforce/wikitext",
    "HuggingFaceM4/WebSight", "RyokoAI/ShareGPT52K", "laion/OIG", "stanfordnlp/SHP", "PleIAs/YouTube-Commons", "Skywork/SkyPile-150B", "glaiveai/glaive-function-calling-v2", "Samsung/samsum", "lmsys/chatbot_arena_conversations", "openbmb/UltraFeedback",
    "lambdalabs/pokemon-blip-captions", "shibing624/medical", "berkeley-nest/Nectar", "Intel/orca_dpo_pairs", "YeungNLP/firefly-train-1.1M", "BAAI/COIG-PC", "meta-math/MetaMathQA", "gsm8k", "codeparrot/github-code", "bookcorpus/bookcorpus",
    "Open-Orca/SlimOrca", "dair-ai/emotion", "CohereForAI/aya_dataset", "legacy-datasets/c4", "cais/mmlu", "open-web-math/open-web-math", "code-search-net/code_search_net", "allenai/WildChat-1M", "rajpurkar/squad", "litagin/moe-speech",
    "Lin-Chen/ShareGPT4V", "shareAI/ShareGPT-Chinese-English-90k", "nomic-ai/gpt4all-j-prompt-generations", "ceval/ceval-exam", "google/fleurs", "openai/webgpt_comparisons", "bigcode/the-stack-v2", "HuggingFaceM4/the_cauldron", "Salesforce/dialogstudio", "LDJnr/Capybara",
    "stanfordnlp/imdb", "nampdn-ai/tiny-codes", "CausalLM/Refined-Anime-Text", "bigscience/P3", "vicgalle/alpaca-gpt4", "bigcode/ta-prompt", "Locutusque/UltraTextbooks", "allenai/c4", "pile-of-law/pile-of-law", "teknium/openhermes",
    "TIGER-Lab/MathInstruct", "HuggingFaceH4/ultrafeedback_binarized", "PygmalionAI/PIPPA", "openai/openai_humaneval", "abisee/cnn_dailymail", "yizhongw/self_instruct", "SirNeural/flan_v2", "nvidia/HelpSteer", "THUDM/AgentInstruct", "nvidia/OpenMathInstruct-1",
    "openai/summarize_from_feedback", "nickrosh/Evol-Instruct-Code-80k-v1", "storytracer/US-PD-Books", "OpenAssistant/oasst2", "Cohere/wikipedia-2023-11-embed-multilingual-v3", "argilla/OpenHermesPreferences", "Hello-SimpleAI/HC3", "SciPhi/textbooks-are-all-you-need-lite", "vikp/textbook_quality_programming", "financial_phrasebank",
    "truthfulqa/truthful_qa", "GAIR/MathPile", "Anthropic/persuasion", "m-a-p/Code-Feedback", "laion/laion2B-en", "wangrui6/Zhihu-KOL", "openchat/openchat_sharegpt4_dataset", "oscar-corpus/oscar", "sahil2801/CodeAlpaca-20k", "Tele-AI/TeleChat-PTD",
    "mozilla-foundation/common_voice_11_0", "mlabonne/orpo-dpo-mix-40k", "Open-Orca/FLAN", "rajpurkar/squad_v2", "nyanko7/LLaMA-65B", "aps/super_glue", "cognitivecomputations/wizard_vicuna_70k_unfiltered", "Amod/mental_health_counseling_conversations", "EleutherAI/proof-pile-2", "ProGamerGov/StableDiffusion-v1-5-Regularization-Images",
    "defunct-datasets/the_pile_books3", "mc4", "knkarthick/dialogsum", "argilla/distilabel-capybara-dpo-7k-binarized", "nyanko7/danbooru2023", "Hello-SimpleAI/HC3-Chinese", "MMMU/MMMU", "ise-uiuc/Magicoder-Evol-Instruct-110K", "argilla/distilabel-intel-orca-dpo-pairs", "H-D-T/Buzz",
    "theblackcat102/evol-codealpaca-v1", "animelover/danbooru2022", "CohereForAI/aya_collection", "allenai/soda", "lvwerra/stack-exchange-paired", "teknium/GPT4-LLM-Cleaned", "BelleGroup/train_1M_CN", "allenai/peS2o", "vivym/midjourney-messages", "oscar-corpus/OSCAR-2301",
    "taesiri/arxiv_qa", "unalignment/toxic-dpo-v0.1", "math-ai/AutoMathText", "mozilla-foundation/common_voice_13_0", "nampdn-ai/tiny-textbooks", "ise-uiuc/Magicoder-OSS-Instruct-75K", "legacy-datasets/common_voice", "armanc/scientific_papers", "mlabonne/guanaco-llama2-1k", "DIBT/10k_prompts_ranked",
    "medical_dialog", "nomic-ai/gpt4all_prompt_generations", "go_emotions", "iamtarun/python_code_instructions_18k_alpaca", "argilla/dpo-mix-7k", "MBZUAI/LaMini-instruction", "qiaojin/PubMedQA", "LinkSoul/instruction_merge_set", "LooksJuicy/ruozhiba", "pleisto/wikipedia-cn-20230720-filtered",
    "kakaobrain/coyo-700m", "gaia-benchmark/GAIA", "PleIAs/Post-OCR-Correction", "fancyzhx/ag_news", "cognitivecomputations/WizardLM_alpaca_evol_instruct_70k_unfiltered", "BelleGroup/train_3.5M_CN", "togethercomputer/Long-Data-Collections", "derek-thomas/ScienceQA", "HuggingFaceM4/OBELICS", "abacusai/SystemChat",
    "google/MusicCaps", "dell-research-harvard/AmericanStories", "shahules786/orca-chat", "li2017dailydialog/daily_dialog", "cognitivecomputations/samantha-data", "allenai/MADLAD-400", "pixparse/idl-wds", "eriktks/conll2003", "oscar-corpus/OSCAR-2201", "BelleGroup/multiturn_chat_0.8M",
    "knowrohit07/know_sql", "bigscience/xP3", "mosaicml/dolly_hhrlhf", "nvidia/ChatQA-Training-Data", "zzliang/GRIT", "cardiffnlp/tweet_eval", "togethercomputer/RedPajama-Data-1T-Sample", "izumi-lab/llm-japanese-dataset", "TigerResearch/pretrain_zh", "Dahoas/rm-static",
    "HuggingFaceH4/stack-exchange-preferences", "hakurei/open-instruct-v1", "liuhaotian/LLaVA-Pretrain", "MMInstruction/M3IT", "lmsys/toxic-chat", "librispeech_asr", "codeparrot/apps", "BelleGroup/train_2M_CN", "laion/gpt4v-dataset", "jondurbin/truthy-dpo-v0.1",
    "argilla/ultrafeedback-binarized-preferences-cleaned", "mbpp", "xlangai/spider", "Helsinki-NLP/opus-100", "openlifescienceai/medmcqa", "BelleGroup/train_0.5M_CN", "defunct-datasets/amazon_reviews_multi", "JeanKaddour/minipile", "michaelwzhu/ChatMed_Consult_Dataset", "MBZUAI/Bactrian-X",
    "allenai/prosocial-dialog", "csebuetnlp/xlsum", "silk-road/Wizard-LM-Chinese-instruct-evol", "allenai/WildChat", "migtissera/Synthia-v1.3", "MarkrAI/KoCommercial-Dataset", "allenai/nllb", "prometheus-eval/Feedback-Collection", "TIGER-Lab/MMLU-Pro", "codeparrot/github-code-clean",
    "zhengyun21/PMC-Patients", "ikala/tmmluplus", "hendrycks/competition_math", "espnet/yodas", "m-a-p/CodeFeedback-Filtered-Instruction", "LDJnr/Puffin", "epfl-llm/guidelines", "maywell/korean_textbooks", "sentence-transformers/embedding-training-data", "huggan/wikiart",
    "Chinese-Vicuna/guanaco_belle_merge_v1.0", "fnlp/moss-002-sft-data", "openbmb/UltraInteract_sft", "allenai/ai2_arc", "deepmind/code_contests", "succinctly/midjourney-prompts", "AI4Math/MathVista", "satellogic/EarthView", "pixparse/pdfa-eng-wds", "BelleGroup/school_math_0.25M",
    "kaist-ai/CoT-Collection", "allenai/objaverse-xl", "Salesforce/wikisql", "zeroshot/twitter-financial-news-sentiment", "mozilla-foundation/common_voice_17_0", "openbmb/UltraInteract_pair", "microsoft/ms_marco", "unimelb-nlp/wikiann", "google/xtreme", "osunlp/Mind2Web",
    "yys/OpenOrca-Chinese", "unalignment/toxic-dpo-v0.2", "nampdn-ai/tiny-strange-textbooks", "empathetic_dialogues", "philschmid/sharegpt-raw", "X2FD/LVIS-Instruct4V", "math_dataset", "sunzeyeah/chinese_chatgpt_corpus", "wanng/midjourney-v5-202304-clean", "ybisk/piqa",
    "IlyaGusev/gpt_roleplay_realm", "cognitivecomputations/Dolphin-2.9", "allenai/sciq", "camel-ai/math", "liuhaotian/LLaVA-CC3M-Pretrain-595K", "silk-road/alpaca-data-gpt4-chinese", "facebook/belebele", "open-phi/textbooks", "SciPhi/AgentSearch-V1", "ylecun/mnist",
    "Yelp/yelp_review_full", "facebook/winoground", "lmsys/mt_bench_human_judgments", "shibing624/sharegpt_gpt4", "gbharti/finance-alpaca", "allenai/tulu-v2-sft-mixture", "andersonbcdefg/synthetic_retrieval_tasks", "Sao10K/Claude-3-Opus-Instruct-15K", "m-a-p/Matrix", "ncbi/pubmed",
    "monology/pile-uncopyrighted", "Open-Orca/SlimOrca-Dedup", "medalpaca/medical_meadow_medqa", "zxbsmk/webnovel_cn", "BI55/MedText", "Rowan/hellaswag", "PKU-Alignment/PKU-SafeRLHF", "rubend18/ChatGPT-Jailbreak-Prompts", "flytech/python-codes-25k", "hollyyfc/tidytuesday_for_python",
    "shibing624/alpaca-zh", "THUDM/LongBench", "glaiveai/glaive-code-assistant", "keivalya/MedQuad-MedicalQnADataset", "arxiv-community/arxiv_dataset", "nyu-mll/multi_nli", "kunishou/databricks-dolly-15k-ja", "lemonilia/LimaRP", "math_qa", "stanfordnlp/sst2",
    "EleutherAI/the_pile_deduplicated", "HuggingFaceH4/CodeAlpaca_20K", "pankajmathur/WizardLM_Orca", "glaiveai/glaive-function-calling", "LDJnr/Pure-Dove", "vikhyatk/lnqa", "hiyouga/DPO-En-Zh-20k", "yfszzx/inspiration", "Dahoas/full-hh-rlhf", "codefuse-ai/Evol-instruction-66k",
    "ZenMoore/RoleBench", "speechcolab/gigaspeech", "neural-bridge/rag-dataset-12000", "defunct-datasets/amazon_us_reviews", "wikimedia/wikisource", "THUDM/humaneval-x", "liyucheng/zhihu_rlhf_3k", "PatronusAI/financebench", "EdinburghNLP/xsum", "unicamp-dl/mmarco",
    "0xJustin/Dungeons-and-Diffusion", "tiange/Cap3D", "NumbersStation/NSText2SQL", "b3x0m/Chinese-H-Novels", "hotpot_qa", "YeungNLP/moss-003-sft-data", "osunlp/MagicBrush", "Yukang/LongAlpaca-12k", "math-ai/StackMathQA", "PolyAI/minds14",
    "FreedomIntelligence/HuatuoGPT-sft-data-v1", "nlpai-lab/kullm-v2", "ai4privacy/pii-masking-200k", "argilla/OpenHermes2.5-dpo-binarized-alpha", "ArmelR/stack-exchange-instruction", "argilla/distilabel-math-preference-dpo", "allenai/openbookqa", "facebook/voxpopuli", "IlyaGusev/ru_turbo_alpaca", "griffin/chain_of_density",
    "jondurbin/gutenberg-dpo-v0.1", "PleIAs/French-PD-Newspapers", "ParlAI/blended_skill_talk", "mandarjoshi/trivia_qa", "ranjaykrishna/visual_genome", "JanosAudran/financial-reports-sec", "fnlp/moss-003-sft-data", "approximatelabs/tablib-v1-full", "mozilla-foundation/common_voice_16_0", "xai-org/RealworldQA",
    "lmsys/lmsys-arena-human-preference-55k", "Abirate/english_quotes", "BelleGroup/generated_chat_0.4M", "maharshipandya/spotify-tracks-dataset", "TokenBender/code_instructions_122k_alpaca_style", "Flmc/DISC-Med-SFT", "ShengbinYue/DISC-Law-SFT", "argilla/ultrafeedback-binarized-preferences", "alexfabbri/multi_news", "nguha/legalbench",
    "Squish42/bluemoon-fandom-1-1-rp-cleaned", "gorilla-llm/APIBench", "OpenAssistant/oasst_top1_2023-08-25", "joujiboi/japanese-anime-speech", "BAAI/CCI-Data", "google-research-datasets/conceptual_captions", "selfrag/selfrag_train_data", "MLCommons/peoples_speech", "laion/laion-coco", "gamino/wiki_medical_terms",
    "yitingxie/rlhf-reward-datasets", "PKU-Alignment/PKU-SafeRLHF-10K", "graelo/wikipedia", "bitext/Bitext-customer-support-llm-chatbot-training-dataset", "AdaptLLM/finance-tasks", "XzJosh/audiodataset", "BAAI/TACO", "nvidia/ChatRAG-Bench", "google/boolq", "kdexd/red_caps",
    "ccdv/pubmed-summarization", "ctheodoris/Genecorpus-30M", "Cohere/wikipedia-22-12-en-embeddings", "tasksource/bigbench", "junelee/sharegpt_deepl_ko", "elyza/ELYZA-tasks-100", "codefuse-ai/CodeExercise-Python-27k", "FreedomIntelligence/ALLaVA-4V", "NilanE/ParallelFiction-Ja_En-100k", "facebook/multilingual_librispeech",
    "ms903/sovits4.0-768vec-layer12", "CohereForAI/xP3x", "princeton-nlp/SWE-bench", "allenai/ultrafeedback_binarized_cleaned", "sujet-ai/Sujet-Finance-Instruct-177k", "tau/commonsense_qa", "ccdv/arxiv-summarization", "AmazonScience/massive", "ShapeNet/ShapeNetCore", "bigbio/med_qa",
    "Cohere/wikipedia-22-12-simple-embeddings", "lukaemon/mmlu", "bigcode/humanevalpack", "ArtifactAI/arxiv-math-instruct-50k", "dikw/hh_rlhf_cn", "food101", "allenai/qasper", "stanfordnlp/snli", "Helsinki-NLP/tatoeba_mt", "laion/laion-high-resolution",
    "facebook/flores", "reazon-research/reazonspeech", "swype/instruct", "athirdpath/DPO_Pairs-Roleplay-Alpaca-NSFW", "cognitivecomputations/dolphin-coder", "McGill-NLP/WebLINX", "sarvamai/samvaad-hi-v1", "froggeric/creativity", "0-hero/Matter-0.1", "NortheasternUniversity/big_patent",
    "statmt/cc100", "jhu-clsp/jfleg", "neulab/conala", "jmhessel/newyorker_caption_contest", "HuggingFace-CN-community/translation", "bigcode/commitpack", "akoksal/LongForm", "JourneyDB/JourneyDB", "OpenGVLab/InternVid", "heliosbrahma/mental_health_chatbot_dataset",
    "mlsum", "google/xtreme_s", "Linaqruf/pixiv-niji-journey", "THUDM/webglm-qa", "starmpcc/Asclepius-Synthetic-Clinical-Notes", "fondant-ai/fondant-cc-25m", "jondurbin/airoboros-3.1", "wenge-research/yayi2_pretrain_data", "TuringsSolutions/NYTWritingStyleGuide", "KBlueLeaf/danbooru2023-sqlite",
    "xx103/NYC_Motor_Vehicle_Collisions_and_Weather_Dataset", "bigcode/self-oss-instruct-sc2-exec-filter-50k", "google-research-datasets/natural_questions", "Helsinki-NLP/open_subtitles", "Dahoas/synthetic-instruct-gptj-pairwise", "open-llm-leaderboard/results", "teknium/trismegistus-project", "ro-h/regulatory_comments", "ibrahimhamamci/CT-RATE", "ruslanmv/ai-medical-chatbot",
    "eli5", "cimec/lambada", "PhilipMay/stsb_multi_mt", "GEM/wiki_lingua", "euirim/goodwiki", "laion/220k-GPT4Vision-captions-from-LIVIS", "sc890/DEEPFRUlT_DATASET", "Replete-AI/code_bagel", "uoft-cs/cifar10", "medical_questions_pairs",
    "codeparrot/codeparrot-clean", "google/bigbench", "camel-ai/physics", "bigcode/commitpackft", "silk-road/ChatHaruhi-54K-Role-Playing-Dialogue", "clouditera/security-paper-datasets", "openerotica/freedom-rp", "Major-TOM/Core-S2L2A", "vblagoje/cc_news", "kilt_tasks",
    "deepmind/pg19", "allenai/winogrande", "aharley/rvl_cdip", "naver-clova-ix/cord-v2", "jamescalam/unsplash-25k-photos", "jkhedri/psychology-dataset", "grammarly/coedit", "Duxiaoman-DI/FinCorpus", "a686d380/h-corpus-2023", "teknium/dataforge-economics",
    "jondurbin/cinematika-v0.1", "mlabonne/chatml_dpo_pairs", "hieunguyenminh/roleplay", "xz56/react-llama", "TeraflopAI/Caselaw_Access_Project", "coastalcph/lex_glue", "cornell-movie-review-data/rotten_tomatoes", "community-datasets/yahoo_answers_topics", "miracl/miracl", "humarin/chatgpt-paraphrases",
    "junelee/wizard_vicuna_70k", "csitfun/LogiCoT", "haonan-li/cmmlu", "shahules786/orca-best", "yuvalkirstain/pickapic_v2", "mozilla-foundation/common_voice_16_1", "Locutusque/UltraTextbooks-2.0", "m-a-p/MAP-CC", "google/code_x_glue_ct_code_to_text", "kmfoda/booksum",
    "hoskinson-center/proof-pile", "kaiokendev/SuperCOT-dataset", "tatsu-lab/alpaca_eval", "kwaikeg/KAgentInstruct", "MaziyarPanahi/WizardLM_evol_instruct_V2_196k", "facebook/xnli", "Muennighoff/flan", "qwedsacf/grade-school-math-instructions", "rickRossie/bluemoon_roleplay_chat_data_300k_messages", "codeparrot/self-instruct-starcoder",
    "umarbutler/open-australian-legal-corpus", "teleprint-me/phi-1", "google/dreambooth", "LDJnr/LessWrong-Amplify-Instruct", "ro-h/regulatory_comments_api", "Severian/Internal-Knowledge-Map", "lamini/earnings-calls-qa", "LanguageBind/Open-Sora-Plan-v1.0.0", "stanfordnlp/coqa", "allenai/ropes",
    "ought/raft", "transformersbook/codeparrot", "nateraw/parti-prompts", "allenai/real-toxicity-prompts", "Muennighoff/natural-instructions", "argilla/databricks-dolly-15k-curated-multilingual", "alpindale/visual-novels", "Norquinal/claude_multiround_chat_30k", "yentinglin/TaiwanChat", "qgyd2021/chinese_ner_sft",
    "LDJnr/Verified-Camel", "WenhaoWang/VidProM", "bigcode/the-stack-v2-dedup", "Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary", "internlm/Agent-FLAN", "isidentical/moondream2-coyo-5M-captions", "fashion_mnist", "shibing624/nli_zh", "Monash-University/monash_tsf", "camel-ai/ai_society",
    "michaelwzhu/ShenNong_TCM_Dataset", "linhtran92/viet_bud500", "Clinton/Text-to-sql-v1", "glaiveai/glaive-code-assistant-v2", "llmware/rag_instruct_benchmark_tester", "jovianzm/Pexels-400k", "WhiteRabbitNeo/WRN-Chapter-1", "Locutusque/function-calling-chatml", "ShimizuYuki/Marvel_network", "clips/mqa",
    "toxigen/toxigen-data", "joelniklaus/Multi_Legal_Pile", "miracl/miracl-corpus", "alespalla/chatbot_instruction_prompts", "teknium/GPTeacher-General-Instruct", "jondurbin/airoboros-gpt4-1.4.1", "VMware/open-instruct", "allenai/reward-bench", "davanstrien/haiku_dpo", "klue",
    "ncbi/ncbi_disease", "esdurmus/wiki_lingua", "wikimedia/wit_base", "shunk031/JGLUE", "llm-wizard/alpaca-gpt4-data-zh", "Vision-CAIR/cc_sbu_align", "pharaouk/dharma-1", "jondurbin/airoboros-2.2.1", "Vezora/Tested-22k-Python-Alpaca", "HAERAE-HUB/KMMLU",
    "MMInstruction/ArxivCap", "jondurbin/py-dpo-v0.1", "PleIAs/French-PD-Books", "CohereForAI/aya_evaluation_suite", "CohereForAI/aya_collection_language_split", "ClusterlabAi/101_billion_arabic_words_dataset", "google/imageinwords", "fancyzhx/amazon_polarity", "ehovy/race", "oscar-corpus/OSCAR-2109",
    "zh-plus/tiny-imagenet", "MoritzLaurer/multilingual-NLI-26lang-2mil7", "tyqiangz/multilingual-sentiments", "detection-datasets/fashionpedia", "EleutherAI/lambada_openai", "Anthropic/model-written-evals", "ds4sd/DocLayNet", "Zellic/smart-contract-fiesta", "FreedomIntelligence/huatuo_encyclopedia_qa", "Chinese-Vicuna/instruct_chat_50k.jsonl",
    "Trelis/function_calling_extended", "FreedomIntelligence/Evol-Instruct-Chinese-GPT4", "Anthropic/discrim-eval", "nlpie/Llama2-MedTuned-Instructions", "PixArt-alpha/SAM-LLaVA-Captions10M", "AkitoP/Hscene-Speech", "mlqa", "webis/tldr-17", "CogComp/trec", "biglam/europeana_newspapers",
    "pacovaldez/stackoverflow-questions", "TigerResearch/sft_zh", "zjunlp/Mol-Instructions", "pufanyi/MIMICIT", "BAAI/JudgeLM-100K", "Trelis/function_calling_v3", "google/Synthetic-Persona-Chat", "FarReelAILab/Machine_Mindset_MBTI_dataset", "jtatman/stable-diffusion-prompts-stats-full-uncensored", "KBlueLeaf/danbooru2023-webp-4Mpixel",
    "THUDM/LongAlign-10k", "LeoZhangzaolin/Graptoloidea-Specimens-Imaging", "ResplendentAI/NSFW_RP_Format_DPO", "RekaAI/VibeEval", "tomg-group-umd/cinepile", "legacy-datasets/banking77", "rmyeid/polyglot_ner", "community-datasets/tapaco", "deepset/germanquad", "laion/laion2B-multi",
    "huggan/smithsonian_butterflies_subset", "CShorten/ML-ArXiv-Papers", "codeparrot/xlcost-text-to-code", "lukaemon/bbh", "thu-coai/Safety-Prompts", "IDEA-CCNL/Ziya-Eval-Chinese", "cognitivecomputations/WizardLM_evol_instruct_V2_196k_unfiltered_merged_split", "beyond/rlhf-reward-single-round-trans_chinese", "jerryjalapeno/nart-100k-synthetic", "vikp/pypi_clean",
    "cognitivecomputations/ultrachat-uncensored", "facebook/emu_edit_test_set", "playgroundai/MJHQ-30K", "zwn22/NC_Crime", "Shitao/MLDR", "Sayali9141/traffic_signal_images", "deutsche-telekom/Ger-RAG-eval", "FiscalNote/billsum", "clue/clue", "theatticusproject/cuad-qa",
    "Helsinki-NLP/opus_books", "SLPL/naab", "Cohere/wikipedia-22-12", "MohamedRashad/ChatGPT-prompts", "HuggingFace-CN-community/Diffusion-book-cn", "HuggingFaceH4/instruction-dataset", "deepset/prompt-injections", "OpenLeecher/Teatime", "math-eval/TAL-SCQ5K", "HackerNoon/tech-company-news-data-dump",
    "LLM360/AmberDatasets", "peiyi9979/Math-Shepherd", "Crystalcareai/MoD", "papluca/language-identification", "bigcode/the-stack-smol", "argilla/news-summary", "CarperAI/openai_summarize_comparisons", "argilla/databricks-dolly-15k-curated-en", "mikex86/stackoverflow-posts", "Anthropic/llm_global_opinions",
    "akjindal53244/Arithmo-Data", "OpenLLM-France/Claire-Dialogue-French-0.1", "arbml/CIDAR", "snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset", "PleIAs/US-PD-Newspapers", "yh0701/FracAtlas_dataset", "somosnlp/Reglamento_Aeronautico_Colombiano_2024GemmaQA", "omi-health/medical-dialogue-to-soap-summary", "argilla/Capybara-Preferences", "UCLNLP/adversarial_qa",
    "convai-challenge/conv_ai_2", "ccdv/govreport-summarization", "mozilla-foundation/common_voice_8_0", "nomic-ai/gpt4all_prompt_generations_with_p3", "hugfaceguy0001/retarded_bar", "lksy/ru_instruct_gpt4", "Linly-AI/Chinese-pretraining-dataset", "mosaicml/instruct-v3", "corbt/all-recipes", "VatsaDev/TinyText",
    "google/docci", "linux-cn/archive", "Johnnyeee/Yelpdata_663", "HuggingFaceTB/cosmopedia-100k", "nyu-mll/blimp", "defunct-datasets/bookcorpusopen", "iwslt2017", "mbien/recipe_nlg", "Helsinki-NLP/tatoeba", "GEM/viggo",
    "bavard/personachat_truecased", "segments/sidewalk-semantic", "PolyAI/banking77", "facebook/pmd", "zeroshot/twitter-financial-news-topic", "nuprl/MultiPL-E", "GBaker/MedQA-USMLE-4-options", "camel-ai/code", "merve/turkish_instructions", "tasksource/oasst1_pairwise_rlhf_reward",
    "winddude/reddit_finance_43_250k", "tiedong/goat", "togethercomputer/RedPajama-Data-Instruct", "DKYoon/SlimPajama-6B", "Maxx0/sexting-nsfw-adultconten", "squarelike/OpenOrca-gugugo-ko", "MMInstruction/VLFeedback", "LLaVA-VL/llava-plus-data", "McAuley-Lab/Amazon-Reviews-2023", "Open-Orca/1million-gpt-4",
    "gwenxin/pills_inside_bottles", "keithito/lj_speech", "ontonotes/conll2012_ontonotesv5", "mwritescode/slither-audited-smart-contracts", "bsmock/pubtables-1m", "tasksource/mmlu", "bigcode/bigcode-pii-dataset", "medalpaca/medical_meadow_wikidoc", "P01son/instructions", "ArtifactAI/arxiv-physics-instruct-tune-30k",
    "mrtoy/mobile-ui-design", "nampdn-ai/tiny-orca-textbooks", "kyujinpy/KOpen-platypus", "YeungNLP/firefly-pretrain-dataset", "unalignment/airoboros-2.2", "totally-not-an-llm/EverythingLM-data-V3", "CASIA-LM/ChineseWebText", "NeuralNovel/Neural-DPO", "AI4Math/MathVerse", "ucinlp/drop",
    "gigaword", "CUHK-CSE/wider_face", "microsoft/wiki_qa", "HUPD/hupd", "liweili/c4_200m", "nielsr/funsd-layoutlmv3", "IDEA-CCNL/laion2B-multi-chinese-subset", "dennlinger/eur-lex-sum", "mitclinicalml/clinical-ie", "Matthijs/cmu-arctic-xvectors",
    "FredZhang7/stable-diffusion-prompts-2.47M", "philschmid/flanv2", "NTU-NLP-sg/xCodeEval", "MadVoyager/stable_diffusion_instructional_dataset", "zetavg/ShareGPT-Processed", "shibing624/nli-zh-all", "oscar-corpus/colossal-oscar-1.0", "greengerong/leetcode", "ProgramComputer/voxceleb", "allenai/paloma",
    "jondurbin/airoboros-3.2", "facebook/anli", "ibm/duorc", "gem", "peluz/lener_br", "Helsinki-NLP/news_commentary", "google-research-datasets/paws-x", "clips/mfaq", "skytnt/anime-segmentation", "alkzar90/NIH-Chest-X-ray-dataset",
    "olm/wikipedia", "jamescalam/youtube-transcriptions", "shjwudp/chinese-c4", "eloukas/edgar-corpus", "reasoning-machines/gsm-hard", "merve/my_notes", "timbrooks/instructpix2pix-clip-filtered", "liswei/rm-static-zhTW", "llm-wizard/alpaca-gpt4-data", "camel-ai/chemistry",
    "THUDM/ImageRewardDB", "rewoo/planner_instruction_tuning_2k", "OpenLeecher/GPT4-10k", "breadlicker45/bread-midi-dataset", "Tarklanse/Traditional_Chinese_roleplay_chat_Dataset", "jat-project/jat-dataset", "lavita/ChatDoctor-HealthCareMagic-100k", "wuliangfo/Chinese-Pixiv-Novel", "knowrohit07/know_medical_dialogue_v2", "hackaprompt/hackaprompt-dataset",
    "maywell/ko_wikidata_QA", "swechatelangana/chandamama-kathalu", "Idavidrein/gpqa", "HuggingFaceH4/deita-10k-v0-sft", "m-a-p/CMMMU", "dcayton/nba_tracking_data_15_16", "kunishou/J-ResearchCorpus", "FreedomIntelligence/ApolloCorpus", "lightblue/tagengo-gpt4", "jojo0217/korean_safe_conversation",
    "hfl/ruozhiba_gpt4_turbo", "deepmind/narrativeqa", "RussianNLP/russian_super_glue", "google/speech_commands", "karpathy/tiny_shakespeare", "facebook/wiki_dpr", "skt/kobest_v1", "laion/laion-art", "gigant/oldbookillustrations", "ontocord/OIG-moderation",
    "cryscan/multilingual-share", "roneneldan/TinyStoriesInstruct", "hltcoe/megawika", "Aeala/ShareGPT_Vicuna_unfiltered", "64bits/lima_vicuna_format", "nampdn-ai/tiny-webtext", "BAAI/COIG-PC-Lite", "LinkSoul/Chinese-LLaVA-Vision-Instructions", "AdaptLLM/medicine-tasks", "MBZUAI/VideoInstruct-100K",
    "jondurbin/contextual-dpo-v0.1", "matlok/multimodal-python-copilot-training-overview", "bai-roleplay/evol-character-200", "cathw/reddit_climate_comment", "wenbopan/Chinese-dpo-pairs", "AI-Lab-Makerere/beans", "indonlp/indonlu", "coastalcph/multi_eurlex", "s3prl/superb", "universal-dependencies/universal_dependencies",
    "Babelscape/wikineural", "pmc/open_access", "winvoker/turkish-sentiment-analysis-dataset", "edinburghcstr/ami", "Erythrocyte/Genshin_Datasets", "bigcode/the-stack-github-issues", "shibing624/CSC", "mattmdjaga/human_parsing_dataset", "camel-ai/biology", "hssd/hssd-hab",
    "PKU-Alignment/BeaverTails", "rhasspy/piper-checkpoints", "visheratin/laion-coco-nllb", "iamtarun/code_instructions_120k_alpaca", "rombodawg/LosslessMegaCodeTrainingV3_1.6m_Evol", "vivym/midjourney-prompts", "qgyd2021/few_shot_intent_sft", "QuyenAnhDE/Diseases_Symptoms", "ajibawa-2023/Python-Code-23k-ShareGPT", "m-a-p/COIG-Kun",
    "CausalLM/GPT-4-Self-Instruct-German", "shareAI/novelai3", "MinervaAI/Aesir-Preview", "wintercoming6/artwork_for_sdxl", "Salesforce/lotsa_data", "ForzaJuve1/UEFA_Euro_2020_Data", "mo-mittal/reddit_political_subs", "Targoman/TLPC", "google-research-datasets/paws", "Stanford/web_questions",
    "bigscience-data/roots_zh-cn_wikipedia", "laion/laion2B-en-aesthetic", "daekeun-ml/naver-news-summarization-ko", "CarperAI/openai_summarize_tldr", "competitions/aiornot", "huggingface/badges", "allenai/lila", "yuvalkirstain/pickapic_v1", "tatsu-lab/alpaca_farm", "cognitivecomputations/open-instruct-uncensored",
    "CheshireAI/guanaco-unchained", "openchat/openchat_sharegpt_v3", "LinkSoul/LLaSM-Audio-Instructions", "totally-not-an-llm/EverythingLM-data-V2", "jinaai/code_exercises", "0-hero/prompt-perfect", "jamescalam/ai-arxiv-chunked", "maywell/ko_Ultrafeedback_binarized", "keirp/hungarian_national_hs_finals_exam", "laion/laion-pop",
    "gvecchio/MatSynth", "baobab-trees/wikipedia-human-retrieval-ja", "mii-llm/gazzetta-ufficiale", "shachardon/ShareLM", "MohamedRashad/midjourney-detailed-prompts", "ade-benchmark-corpus/ade_corpus_v2", "uoft-cs/cifar100", "mhardalov/exams", "josecannete/large_spanish_corpus", "allenai/quac",
    "microsoft/xglue", "huggingface/documentation-images", "seamew/ChnSentiCorp", "tau/scrolls", "bible-nlp/biblenlp-corpus", "JulesBelveze/tldr_news", "christopher/rosetta-code", "inria-soda/tabular-benchmark", "beyond/chinese_clean_passages_80m", "bigbio/pubmed_qa",
    "Cohere/miracl-zh-queries-22-12", "koutch/stackoverflow_python", "ACCA225/Kaggle-Stable-Diffusion", "Yasbok/Alpaca_arabic_instruct", "bertin-project/alpaca-spanish", "laion/laion400m", "axiong/pmc_oa", "medalpaca/medical_meadow_medical_flashcards", "dominguesm/Canarim-Instruct-PTBR-Dataset", "p1atdev/niji-v5",
    "zetavg/coct-en-zh-tw-translations-twp-300k", "skeskinen/TinyStories-GPT4", "xmcmic/PMC-VQA", "beomi/KoAlpaca-v1.1a", "ecnu-icalk/educhat-sft-002-data-osm", "kyujinpy/OpenOrca-KO", "open-phi/programming_books_llama", "hkust-nlp/deita-10k-v0", "jxu124/OpenX-Embodiment", "m-a-p/MusicPile",
    "ajibawa-2023/Code-290k-ShareGPT", "bai-roleplay/evol-character-entire", "minhanhto09/NuCLS_dataset", "cl-nagoya/auto-wiki-qa", "speechbrain/common_language", "ucirvine/sms_spam", "Babelscape/rebel-dataset", "cfilt/iitb-english-hindi", "gfissore/arxiv-abstracts-2021", "mozilla-foundation/common_voice_7_0",
    "sil-ai/bloom-lm", "kensho/spgispeech", "bigscience/xP3all", "llm-wizard/dolly-15k-instruction-alpaca-format", "liyucheng/zhihu_26k", "tarungupta83/MidJourney_v5_Prompt_dataset", "jondurbin/airoboros-uncensored", "llm-blender/mix-instruct", "UmaDiffusion/ULTIMA", "BAAI/SVIT",
    "AdiOO7/llama-2-finance", "togethercomputer/llama-instruct", "kingbri/PIPPA-shareGPT", "Minami-su/roleplay_multiturn_chat_1k_zh_v0.1", "Illia56/Military-Aircraft-Detection", "cis-lmu/Glot500", "facebook/emu_edit_test_set_generations", "Yukang/LongAlpaca-16k-length", "THUDM/CogVLM-SFT-311K", "qnguyen3/llava-fn-calling",
    "Locutusque/hercules-v2.0", "HathawayLiu/housing_dataset", "bigcode/the-stack-v2-train-full-ids", "YXu120/NC_Education", "motherduckdb/duckdb-text2sql-25k", "Wenetspeech4TTS/WenetSpeech4TTS", "naklecha/minecraft-question-answer-700k", "HannahRoseKirk/prism-alignment", "halabi2016/arabic_speech_corpus", "allenai/common_gen",
    "health_fact", "pfb30/multi_woz_v22", "nfL6/yahoo_answers_qa", "MLCommons/ml_spoken_words", "ucberkeley-dlab/measuring-hate-speech", "bigscience/xP3mt", "sayakpaul/nyu_depth_v2", "argilla/medical-domain", "nlphuji/flickr30k", "aadityaubhat/GPT-wiki-intro",
    "nbertagnolli/counsel-chat", "theblackcat102/codex-math-qa", "RyokoAI/Syosetu711K", "emre/stanford-alpaca-cleaned-turkish-translated", "somosnlp-hackathon-2023/Habilidades_Agente_v1", "recastai/LAION-art-EN-improved-captions", "FreedomIntelligence/huatuo_knowledge_graph_qa", "FreedomIntelligence/ShareGPT-CN", "Mutonix/RefGPT-Fact", "nlpai-lab/databricks-dolly-15k-ko",
    "TempoFunk/webvid-10M", "shinonomelab/cleanvid-15m_map", "smangrul/code-chat-assistant-v1", "OleehyO/latex-formulas", "daat/DATA", "axiong/pmc_llama_instructions", "AdaptLLM/law-tasks", "chargoddard/rpguild", "AiresPucrs/stanford-encyclopedia-philosophy", "amaai-lab/MusicBench",
    "diffusers/pokemon-gpt4-captions", "migtissera/Tess-Coder-v1.0", "HaoyeZhang/RLHF-V-Dataset", "togethercomputer/glaive-function-calling-v2-formatted", "osunlp/TravelPlanner", "BioMistral/BioInstructQA", "misikoff/zillow", "MedRAG/pubmed", "Writer/omniact", "openbmb/UltraSafety",
    "visheratin/realworldqa", "lorinma/ChineseEncyclopedia", "sealuzh/app_reviews", "levow/msra_ner", "openslr/openslr", "INK-USC/riddle_sense", "zhoubolei/scene_parse_150", "allenai/scitldr", "google-research-datasets/tydiqa", "IlyaGusev/gazeta",
    "albertvillanova/legal_contracts", "google-research-datasets/conceptual_12m", "facebook/textvqa", "VIMA/VIMA-Data", "hanamizuki-ai/genshin-voice-v3.3-mandarin", "Nerfgun3/sakimi-chan_LoRA", "cyberagent/crello", "jxm/the_office_lines", "WynterJones/chatgpt-roles", "gbharti/wealth-alpaca_lora",
    "THUIR/T2Ranking", "IlyaGusev/ru_turbo_saiga", "tasksource/ScienceQA_text_only", "cvssp/WavCaps", "lighteval/MATH", "kunishou/oasst1-89k-ja", "zetavg/zh-tw-wikipedia", "lighteval/legal_summarization", "skeskinen/TinyStories-hf", "silk-road/chinese-dolly-15k",
    "TigerResearch/tigerbot-zhihu-zh-10k", "open-llm-leaderboard/requests", "mlabonne/guanaco-llama2", "totally-not-an-llm/EverythingLM-data", "BELLE-2/train_3.5M_CN_With_Category", "rizerphe/glaive-function-calling-v2-llama", "rombodawg/LimitlessMegaCodeTraining", "re-align/just-eval-instruct", "IlyaGusev/pippa_scored", "IGNF/FLAIR",
    "allenai/WildChat-nontoxic", "Unbabel/TowerBlocks-v0.1", "ShoukanLabs/AniSpeech", "unsloth/notebooks", "GAIR/MathPile_Commercial", "abacusai/MetaMathFewshot", "DiscoResearch/germanrag", "cdoswald/SPIDER", "yixuantt/MultiHopRAG", "instructkr/ko_elo_arena_0207",
    "osunlp/SMolInstruct", "allenai/WildBench", "FuseAI/FuseChat-Mixture", "Vezora/Tested-143k-Python-Alpaca", "microsoft/cats_vs_dogs", "tdavidson/hate_speech_offensive", "SNOW-NLP/snow_simplified_japanese_corpus", "timit-asr/timit_asr", "webnlg-challenge/web_nlg", "michaelauli/wiki_bio",
    "kili-technology/plastic_in_river", "qanastek/MASSIVE", "google/wit", "sil-ai/bloom-speech", "FacePerceiver/laion-face", "codeparrot/codecomplex", "codeparrot/github-jupyter-code-to-text", "neuralworm/stable-diffusion-discord-prompts", "detection-datasets/coco", "Gxg/Math23K",
    "ashraq/fashion-product-images-small", "animelover/genshin-impact-images", "suolyer/webqa", "fusing/fill50k", "dominguesm/alpaca-data-pt-br", "multimodalart/facesyntheticsspigacaptioned", "jiacheng-ye/logiqa-zh", "sam-mosaic/vicuna_alpaca_hc3_chatml", "thefcraft/civitai-stable-diffusion-337k", "Nan-Do/instructional_code-search-net-python",
    "izumi-lab/llm-japanese-dataset-vanilla", "xmj2002/Chinese_modern_classical", "cognitivecomputations/based", "laion/strategic_game_chess", "jondurbin/airoboros-gpt4-1.2", "jondurbin/airoboros-gpt4-m2.0", "rombodawg/LosslessMegaCodeTrainingV2", "shareAI/CodeChat", "qgyd2021/h_novel", "BAAI/COIG-PC-core",
    "Duxiaoman-DI/FinanceIQ", "Unified-Language-Model-Alignment/Anthropic_HH_Golden", "osunlp/TableInstruct", "CollectiveCognition/chats-data-2023-10-16", "hypervariance/function-calling-sharegpt", "google/reveal", "corbyrosset/researchy_questions", "Locutusque/Hercules-v3.0", "jmc255/aphantasia_drawing_dataset", "sayhan/strix-philosophy-qa",
    "fnlp/AnyInstruct", "NousResearch/json-mode-eval", "XintongHe/Stomatal_Images_Datasets", "abacusai/MetaMath_DPO_FewShot", "coseal/CodeUltraFeedback", "BAAI/CCI2-Data", "Astris/LA-Times", "H-D-T/RLSTACK", "deepmind/aqua_rat", "abuelkhair-corpus/arabic_billion_words",
    "google/code_x_glue_tc_text_to_code", "medal", "IWSLT/mt_eng_vietnamese", "quora-competitions/quora", "CSTR-Edinburgh/vctk", "wmt/wmt19", "dalle-mini/YFCC100M_OpenAI_subset", "merve/poetry", "yhavinga/ccmatrix", "silver/personal_dialog",
    "embedding-data/sentence-compression", "mozilla-foundation/common_voice_10_0", "m1guelpf/nouns", "Fazzie/Teyvat", "daspartho/stable-diffusion-prompts", "cardiffnlp/tweet_sentiment_multilingual", "PublicPrompts/Karsh", "MCG-NJU/MultiSports", "Dahoas/static-hh", "CarperAI/pilev2-dev",
    "shibing624/AdvertiseGen", "andersonbcdefg/supernatural-instructions-2m", "azcorpus/azcorpus_v0", "cognitivecomputations/oa_leet10k", "Abrumu/Fashion_controlnet_dataset_V3", "tasksource/tasksource-instruct-v0", "wenge-research/yayi_domain_subset", "ignmilton/ign_clean_instruct_dataset_500k", "changpt/ko-lima-vicuna", "pankajmathur/alpaca_orca",
    "marhensa/comfyui-workflow", "jondurbin/airoboros-2.1", "M-A-D/Mixed-Arabic-Datasets-Repo", "taide/TAIDE-14-tasks", "manu/project_gutenberg", "Lakera/gandalf_ignore_instructions", "goendalf666/sales-conversations", "yuyijiong/Multi-Doc-QA-Chinese", "fnlp/character-llm-data", "wenge-research/yayi_uie_sft_data",
    "glaiveai/glaive-code-assistant-v3", "davidchan/anim400k", "prometheus-eval/Preference-Collection", "numind/NuNER", "YuxuanZhang888/ColonCancerCTDataset", "TIGER-Lab/SKGInstruct", "CyberNative/Code_Vulnerability_Security_DPO", "hiyouga/glaive-function-calling-v2-sharegpt", "ai4bharat/sangraha", "ontocord/viet4all",
    "cloneofsimo/imagenet.int8", "Replete-AI/code_bagel_hermes-2.5", "amirveyseh/acronym_identification", "cornell-movie-dialog/cornell_movie_dialog", "fancyzhx/dbpedia_14", "esnli", "fever", "google/jigsaw_toxicity_pred", "google/xquad", "NbAiLab/NCC",
    "ccdv/cnn_dailymail", "ccdv/patent-classification", "DFKI-SLT/few-nerd", "solomonk/reddit_mental_health_posts", "carolina-c4ai/corpus-carolina", "thu-coai/lccc", "fabiochiu/medium-articles", "FinanceInc/auditor_sentiment", "nateraw/midjourney-texttoimage-new", "HuggingFaceH4/self-instruct-seed",
    "RyokoAI/CNNovel125K", "IndianaUniversityDatasetsModels/MIMIC-medical-report", "samhog/psychology-10k", "HuggingFaceH4/databricks_dolly_15k", "heegyu/open-korean-instructions", "logo-wizard/modern-logo-dataset", "sam-mosaic/hhrlhf_evol_chatml", "4eJIoBek/PAIT-Downloads", "kunishou/hh-rlhf-49k-ja", "fblgit/tree-of-knowledge",
    "TigerResearch/tigerbot-law-plugin", "kaist-ai/Multilingual-CoT-Collection", "mcipriano/stackoverflow-kubernetes-questions", "jondurbin/airoboros-gpt4-1.4", "SALT-NLP/LLaVAR", "declare-lab/flan-mini", "jondurbin/airoboros-gpt4-2.0", "seungheondoh/LP-MusicCaps-MSD", "AILab-CVC/SEED-Bench", "zjunlp/InstructIE",
    "nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions", "nampdn-ai/tiny-lessons", "Healthy13/Text2SQL", "MBZUAI-LLM/SlimPajama-627B-DC", "a686d380/sis-novel", "fedml/PubMedQA_instruction", "meta-math/MetaMathQA-40K", "PocketDoc/Choose-Your-Story-Long-Text-Adventures", "SinKove/synthetic_mammography_csaw", "unalignment/spicy-3.1",
    "locuslab/TOFU", "OpenGVLab/VideoChat2-IT", "LLM360/CrystalCoderDatasets", "argilla/ultrafeedback-curated", "HuggingFaceH4/grok-conversation-harmless", "HuggingFaceH4/OpenHermes-2.5-1k-longest", "Ziyuan111/DurhamTrees", "2A2I/Arabic-OpenHermes-2.5", "Locutusque/arc-cot", "osunlp/Multimodal-Mind2Web",
    "rc9494/SP500_Date_Offset", "EleutherAI/lichess-puzzles", "conceptnet5/conceptnet5", "allenai/cosmos_qa", "thunlp/docred", "md_gender_bias", "mkqa", "iastate/onestop_english", "KorQuAD/squad_kor_v1", "allenai/swag",
    "tweets-hate-speech-detection/tweets_hate_speech_detection", "wmt/wmt16", "ChristophSchuhmann/MS_COCO_2017_URL_TEXT", "SetFit/emotion", "ai4bharat/samanantar", "ccdv/arxiv-classification", "mteb/tweet_sentiment_extraction", "beki/privy", "zoheb/sketch-scene", "WINGNUS/ACL-OCL",
    "haor/pixiv_month_top50", "HuggingFaceM4/COCO", "haor/pixiv-yandere", "Plachta/Umamusume-voice-text-pairs", "keremberke/chest-xray-classification", "keremberke/table-extraction", "silatus/1k_Website_Screenshots_and_Metadata", "IlyaGusev/habr", "KrakExilios/koreandoll", "pmoe7/SP_500_Stocks_Data-ratios_news_price_10_yrs",
    "potsawee/wiki_bio_gpt3_hallucination", "RyokoAI/Fandom23K", "Bingsu/ko_alpaca_data", "medalpaca/medical_meadow_wikidoc_patient_information", "Papersnake/people_daily_news", "FreedomIntelligence/phoenix-sft-data-v1", "howard-hou/OCR-VQA", "silk-road/Vanilla-chinese-alpaca-luotuo", "danielv835/personal_finance_v0.2", "silk-road/Luotuo-QA-A-CoQA-Chinese",
    "gretelai/symptom_to_diagnosis", "agkphysics/AudioSet", "YeungNLP/ultrachat", "Iess/chinese_modern_poetry", "wendlerc/RenderedText", "Oasis-Team/Oasis-Corpus", "qgyd2021/chinese_chitchat", "MattCoddity/dockerNLcommands", "yuyijiong/Long-Instruction", "Skywork/ChineseDomainModelingEval",
    "xinrongzhang2022/InfiniteBench", "MohamedRashad/multilingual-tts", "silk-road/ChatHaruhi-Expand-118K", "Luckyjhg/Geo170K", "andersonbcdefg/synthetic_tuples_gpt35_turbo", "Rtian/DebugBench", "euclaise/reddit-instruct", "Locutusque/hercules-v1.0", "mastergopote44/Long-Term-Care-Aggregated-Data", "ontocord/CulturaY",
    "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M", "mlabonne/chatml-OpenHermes2.5-dpo-binarized-alpha", "jg583/NSynth", "storytracer/LoC-PD-Books", "zhongshsh/CLoT-Oogiri-GO", "davidkim205/kollm-converations", "Locutusque/hercules-v4.0", "tdiggelm/climate_fever", "hfl/cmrc2018", "mrqa-workshop/mrqa",
    "google-research-datasets/nq_open", "kyunghyuncho/search_qa", "IWSLT/ted_talks_iwslt", "ubuntu-dialogs-corpus/ubuntu_dialogs_corpus", "SetFit/enron_spam", "gsarti/flores_101", "vblagoje/lfqa", "huggan/pokemon", "joelniklaus/lextreme", "OxAISH-AL-LLM/wiki_toxic",
    "tomasg25/scientific_lay_summarisation", "svjack/pokemon-blip-captions-en-zh", "lambdalabs/naruto-blip-captions", "shunk031/wrime", "marmal88/skin_cancer", "IlyaGusev/rulm", "datadrivenscience/ship-detection", "Junity/UmaMusume-TokaiTeio-Dataset", "Den4ikAI/russian_dialogues", "LinhDuong/chatdoctor-200k",
    "Nebulous/gpt4all_pruned", "camel-ai/ai_society_translated", "alpindale/light-novels", "iamketan25/roleplay-instructions-dataset", "VMware/open-instruct-v1-oasst-dolly-hhrlhf", "Nan-Do/code-search-net-python", "ShoukanLabs/OpenNiji-Dataset", "Birchlabs/openai-prm800k-stepwise-critic", "Norquinal/claude_evol_instruct_210k", "mlfoundations/datacomp_1b",
    "tasksource/icl-symbol-tuning-instruct", "findnitai/english-to-hinglish", "pankajmathur/dolly-v2_orca", "sudy-super/dialogsum-ja", "sayakpaul/hf-codegen-v2", "FreedomIntelligence/CMB", "jamescalam/llama-2-arxiv-papers-chunked", "smangrul/hf-stack-v1", "abacusai/LongChat-Lines", "PetraAI/PetraAI",
    "sinarashidi/alpaca-persian", "neural-bridge/rag-hallucination-dataset-1000", "google/trueteacher", "twang2218/chinese-law-and-regulations", "Loie/Auto-ACD", "CollectiveCognition/chats-data-2023-09-22", "CollectiveCognition/chats-data-2023-09-27", "a686d380/h-eval", "guangyil/laion-coco-aesthetic", "ajibawa-2023/Code-74k-ShareGPT",
    "ChuckMcSneed/NeoEvalPlusN_benchmark", "matsuxr/JaGovFaqs-22k", "NobodyExistsOnTheInternet/ToxicQAFinal", "jondurbin/bagel-v0.3", "allenai/preference-test-sets", "xingyaoww/code-act", "moukaii/Tuberculosis_Dataset", "abacusai/ARC_DPO_FewShot", "tinyBenchmarks/tinyMMLU", "HPLT/hplt_monolingual_v1_2",
    "maywell/koVast", "unicamp-dl/quati", "YanweiLi/MGM-Instruction", "BLINK-Benchmark/BLINK", "abacusai/SystemChat-1.1", "DLI-Lab/pearl", "Vi-VLM/Vista", "microsoft/crd3", "hate_speech18", "Helsinki-NLP/kde4",
    "kuznetsoffandrey/sberquad", "McGill-NLP/stereoset", "unimorph/universal_morphologies", "uclanlp/wino_bias", "CAiRE/ASCEND", "huggingface/label-files", "laion/laion5B-index", "vicenteor/sbu_captions", "McGill-NLP/FaithDial", "LIUM/tedlium",
    "AlekseyKorshuk/persona-chat", "allenai/multi_lexsum", "DeveloperOats/DBPedia_Classes", "shailja/Verilog_GitHub", "akariasai/PopQA", "deepghs/game_characters", "nlphuji/whoops", "FredZhang7/anime-prompts-180K", "HuggingFaceH4/instruct_me", "mozilla-foundation/common_voice_12_0",
    "LangChainDatasets/agent-search-calculator", "jamescalam/langchain-docs", "cognitivecomputations/leet10k-alpaca", "Babelscape/multinerd", "kz-transformers/multidomain-kazakh-dataset", "LLMs/Alpaca-ShareGPT", "milashkaarshif/MoeGirlPedia_wikitext_raw_archive", "jainr3/diffusiondb-pixelart", "tau/zero_scrolls", "MU-NLPC/Calc-ape210k",
    "dbdu/ShareGPT-74k-ko", "bavest/fin-llama-dataset", "TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k", "Slep/LAION-RVS-Fashion", "flaviagiammarino/vqa-rad", "L4NLP/LEval", "sudy-super/CoTangent", "newsletter/SDXL-Artists", "liuhaotian/llava-bench-in-the-wild", "mlabonne/CodeLlama-2-20k",
    "lamini/lamini_docs", "marmikpandya/mental-health", "ibm-nasa-geospatial/multi-temporal-crop-classification", "Universal-NER/Pile-NER-type", "m720/SHADR", "nampdn-ai/tiny-math-textbooks", "squarelike/ko_medical_chat", "declare-lab/HarmfulQA", "OpenDriveLab/DriveLM", "neovalle/H4rmony",
    "vibhorag101/phr_mental_therapy_dataset", "Vision-Flan/vision-flan_191-task_1k", "ahmed-masry/ChartQA", "ProlificAI/social-reasoning-rlhf", "BAAI/DataOptim", "Heralax/Augmental-Dataset", "LLM-Tuning-Safety/HEx-PHI", "kwaikeg/KAgentBench", "SeaLLMs/Sea-bench", "athirdpath/DPO_Pairs-Roleplay-Alpaca-NSFW-v1-SHUFFLED",
    "yale-nlp/FOLIO", "RealTimeData/bbc_news_alltime", "HuggingFaceH4/orca_dpo_pairs", "NebulaeWis/gelbooru_images", "llm-blender/Unified-Feedback", "grimulkan/LimaRP-augmented", "cyberagent/chatbot-arena-ja-calm2-7b-chat-experimental", "ehristoforu/midjourney-images", "Jiwonny29/project1", "Major-TOM/Core-S2L1C",
    "gorilla-llm/Berkeley-Function-Calling-Leaderboard", "julep-ai/openai-community-posts", "SALT-NLP/Design2Code", "Locutusque/OpenCerebrum-SFT", "m-a-p/CodeEditorBench", "chansung/merged_ds_coding", "spectrallabs/credit-scoring-training-dataset", "shareAI/DPO-zh-en-emoji", "rqq/GLM-4-Instruct-4K-zh", "Helsinki-NLP/bible_para",
    "UFRGS/brwac", "ZihanWangKi/conllpp", "facebook/covost2", "head_qa", "facebook/lama", "yaolu/multi_x_science_sum", "ptb-text-only/ptb_text_only", "allenai/social_bias_frames", "stanfordnlp/sst", "defunct-datasets/the_pile_openwebtext2",
    "google/wiki40b", "google-research-datasets/wiki_atomic_edits", "botisan-ai/cantonese-mandarin-translations", "nlpaueb/finer-139", "Stanford/wikitablequestions", "silver/lccc", "facebook/content_rephrasing", "Twitter/TwitterFollowGraph", "Nerfgun3/wlop_style", "TheFusion21/PokemonCards",
    "jeanlee/kmhas_korean_hate_speech", "sander-wood/irishman", "tobiolatunji/afrispeech-200", "swaption2009/20k-en-zh-translation-pinyin-hsk", "danielshemesh/midjourney", "Elfsong/ClinicalDataset", "Den4ikAI/russian_instructions", "paulofinardi/OIG_small_chip2_portuguese_brasil", "acheong08/nsfw_reddit", "VISION-Workshop/VISION-Datasets",
    "P1ayer-1/chatgpt-conversations-chatlogs.net", "wavpub/JinJinLeDao_QA_Dataset", "lang-uk/every_prompt", "pki/SecurityGPT", "zjkarina/matreshka", "deepghs/nsfw_detect", "JasperLS/prompt-injections", "ccmusic-database/music_genre", "jondurbin/airoboros-gpt4", "TigerResearch/pretrain_en",
    "mit-han-lab/awq-model-zoo", "Nan-Do/reason_code-search-net-python", "saldra/sakura_japanese_dataset", "explodinggradients/fiqa", "64bits/lex_fridman_podcast_for_llm_vicuna", "KShivendu/dbpedia-entities-openai-1M", "Glavin001/startup-interviews", "FredZhang7/toxi-text-3M", "joonhok-exo-ai/korean_law_open_data_precedents", "UmaDiffusion/ULTIMA-prompts",
    "ArtifactAI/arxiv_python_research_code", "NebulaByte/E-Commerce_Customer_Support_Conversations", "HuggingFaceM4/LLaVAR-Instruct-16K", "Locutusque/InstructMix", "shahules786/Multi-chapter-summaries", "ai4privacy/pii-masking-65k", "Universal-NER/Pile-NER-definition", "jojo0217/korean_rlhf_dataset", "kernelmachine/open-license-corpus", "Xilabs/PIPPA-alpaca",
    "Suprit/CMtMedQA", "ticoAg/Chinese-medical-dialogue", "Yirany/UniMM-Chat", "xuqinyang/BaiduBaike-5.63M", "jamescalam/agent-conversations-retrieval-tool", "zhiqings/LLaVA-Human-Preference-10K", "qgyd2021/rlhf_reward_dataset", "gathnex/Gath_baize", "a686d380/h-corpus-raw", "flytech/llama-python-codes-30k",
    "open-phi/ft-sample-mistral", "hkust-nlp/deita-6k-v0", "Doctor-Shotgun/no-robots-sharegpt", "styletts2-community/multilingual-phonemes-10k-alpha", "imone/OpenOrca_FLAN", "osv5m/osv5m", "multimodalart/steamboat-willy-frames", "irlab-udc/metahate", "grimulkan/theory-of-mind", "ai4bharat/indic-instruct-data-v0.1",
    "kobprof/skolegpt-instruct", "Ejafa/ye-pop", "steamcyclone/Pill_Ideologies-Post_Titles", "euclaise/reddit-instruct-curated", "VatsaDev/animebench-alpha", "0-hero/prompt-perfect-dpo", "MedRAG/textbooks", "TIGER-Lab/Mantis-Instruct", "ChuckMcSneed/various_RP_system_prompts", "chenmingxuan/Chinese-Patent-Summary",
    "cassiekang/cub200_dataset", "antiven0m/catboros-3.2-dpo", "ai4privacy/pii-masking-300k", "multilingual/orca_dpo_pairs", "BigAction/the-wave-clean", "legacy-datasets/ami", "TheBritishLibrary/blbooks", "convai-challenge/conv_ai_3", "e2e_nlg", "ethos",
    "Helsinki-NLP/europarl", "hkcancor", "ucsbnlp/liar", "Maluuba/newsqa", "SemEvalWorkshop/sem_eval_2018_task_1", "rcds/swiss_judgment_prediction", "JAugusto97/told-br", "leondz/wnut_17", "CodedotAI/code_clippy_github", "castorini/mr-tydi",
    "flax-sentence-embeddings/stackexchange_math_jsonl", "jfrenz/legalglue", "ml6team/cnn_dailymail_nl", "sentence-transformers/parallel-sentences", "sentence-transformers/reddit-title-body", "stas/openwebtext-10k", "Azu/Handwritten-Mathematical-Expression-Convert-LaTeX", "patriziobellan/PET", "mozilla-foundation/common_voice_9_0", "bloomberg/entsum",
    "carblacac/twitter-sentiment-analysis", "HuggingFaceM4/VQAv2", "LHF/escorpius", "owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH", "masakhane/mafand", "Muennighoff/P3", "Dahoas/instruct-synthetic-prompt-responses", "mjw/stock_market_tweets", "Korakoe/NijiJourney-Prompt-Pairs", "mrm8488/unnatural-instructions-full",
    "yuvalkirstain/PickaPic", "keremberke/blood-cell-object-detection", "keremberke/license-plate-object-detection", "forta/malicious-smart-contract-dataset", "ChristophSchuhmann/essays-with-instructions", "HuggingFaceH4/helpful-instructions", "nanaaaa/emotion_chinese_english", "wbbbbb/pclue", "lansinuote/ChnSentiCorp", "katanaml-org/invoices-donut-data-v1",
    "mxeval/mbxp", "somosnlp/somos-clean-alpaca-es", "amaydle/npc-dialogue", "KK04/LogicInference_OA", "rajuptvs/ecommerce_products_clip", "hanamizuki-ai/genshin-voice-v3.5-mandarin", "sukaka/novelai-webui", "icybee/share_gpt_90k_v1", "michelleyunun/therapydata", "jaydenccc/AI_Storyteller_Dataset",
    "atasoglu/databricks-dolly-15k-tr", "PaulAdversarial/all_news_finance_sm_1h2023", "juletxara/mgsm", "FreedomIntelligence/huatuo26M-testdatasets", "mio/sukasuka-anime-vocal-dataset", "causalnlp/corr2cause", "tabtoyou/KoLLaVA-Instruct-150k", "ibm-nasa-geospatial/hls_burn_scars", "hkust-nlp/felm", "nisaar/Lawyer_GPT_India",
    "mrzlab630/trading-candles", "ai4privacy/pii-masking-43k", "burkelibbey/colors", "SiberiaSoft/SiberianPersonaChat", "abacusai/WikiQA-Free_Form_QA", "LibrAI/do-not-answer", "nampdn-ai/mini-CoT-Collection", "nampdn-ai/devdocs.io", "TokenBender/roleplay_alpaca", "bupt/LawDataset-BUPT",
    "jondurbin/airoboros-2.2", "apf1/datafilteringnetworks_2b", "04RR/tiny-instruct", "emozilla/yarn-train-tokenized-16k-mistral", "FreedomIntelligence/Huatuo26M-Lite", "Hypersniper/riddles_v1", "q-future/Q-Instruct-DB", "ai-forever/MERA", "THUDM/BPO", "echo840/Detailed_Caption",
    "glnmario/news-qa-summarization", "TriadParty/deepsex-RP", "pixparse/cc3m-wds", "Minami-su/Anime_novel_datasets", "Gourieff/ReActor", "cognitivecomputations/Code-74k-ShareGPT-Vicuna", "dataautogpt3/Dalle3", "DL3DV/DL3DV-Benchmark", "CausalLM/GPT-4-Self-Instruct-Turkish", "sablo/oasst2_curated",
    "STEM-AI-mtl/Electrical-engineering", "ikawrakow/imatrix-from-wiki-train", "somewheresystems/dataclysm-arxiv", "fblgit/simple-math", "fblgit/simple-math-DPO", "acon96/Home-Assistant-Requests", "Query-of-CC/Knowledge_Pile", "OpenDatasets/dalle-3-dataset", "ptx0/photo-concept-bucket", "zjunlp/iepile",
    "BatsResearch/ctga-v1", "MMInstruction/ArxivQA", "hotchpotch/JQaRA", "sean0042/KorMedMCQA", "p1atdev/ichikara-instruction", "maywell/LogicKor", "davanstrien/dataset-tldr", "xcodemind/vision2ui", "lawinstruct/lawinstruct", "UCSC-VLAA/HQ-Edit",
    "kigner/ruozhiba-llama3-tt", "H-D-T/Select-Stack", "mutiyama/alt", "iabufarha/ar_sarcasm", "nilc-nlp/assin2", "cam-cst/cbt", "eurlex", "facebook/kilt_wikipedia", "legacy-datasets/multilingual_librispeech", "ucirvine/reuters21578",
    "stanfordnlp/sentiment140", "ccasimiro/squad_es", "defunct-datasets/the_pile_stack_exchange", "facebook/wiki_movies", "Fraser/python-state-changes", "Hellisotherpeople/DebateSum", "SocialGrep/one-million-reddit-jokes", "blinoff/medical_qa_ru_data", "huggingface/transformers-metadata", "indonesian-nlp/id_newspapers_2018",
    "openclimatefix/nimrod-uk-1km", "sentence-transformers/msmarco-hard-negatives", "nthngdy/oscar-small", "jiangjiechen/ekar_chinese", "sil-ai/bloom-captioning", "orieg/elsevier-oa-cc-by", "imagenet_sketch", "sileod/movie_recommendation", "google/quickdraw", "huggingface-legal/takedown-notices",
    "demelin/moral_stories", "RUCAIBox/Chinese-Generation", "Bingsu/zeroth-korean", "shjwudp/shu", "CarperAI/pile-v2-small-filtered", "citeseerx/ACL-fig", "keremberke/painting-style-classification", "jordyvl/DUDE_loader", "mlfoundations/datacomp_pools", "Loie/VGGSound",
    "artem9k/ai-text-detection-pile", "HuggingFaceH4/hhh_alignment", "hendrycks/ethics", "IlyaGusev/pikabu", "Aditya011/autotrain-data-nl-to-sql", "sedthh/tv_dialogue", "AnonymousSub/MedQuAD_Context_Question_Answer_Triples_TWO", "instruction-tuning-sd/cartoonization", "Polyglot-or-Not/Fact-Completion", "llm-wizard/Product-Descriptions-and-Ads",
    "emplocity/owca", "FronkonGames/steam-games-dataset", "lucasmccabe-lmi/codex_math_qa_alpaca_style", "ms903/Diff-SVC-refactor-pre-trained-model", "FourthBrainGenAI/AI-Superstar-Dataset", "Maciel/FinCUGE-Instruction", "HuggingFaceH4/code_evaluation_prompts", "hoskinson-center/minif2f-lean4", "Fsoft-AIC/the-vault-function", "wangrongsheng/HealthCareMagic-100k-en",
    "edarchimbaud/timeseries-1d-stocks", "lighteval/mmlu", "lucasmccabe-lmi/CodeAlpaca-20k", "DavidVivancos/MindBigData2023_MNIST-8B", "Meranti/CLAP_freesound", "flaviagiammarino/path-vqa", "projectlosangeles/Los-Angeles-MIDI-Dataset", "Babelscape/SREDFM", "Norquinal/claude_multi_instruct_1k", "shumpei2525/fine_tuning521k-ja",
    "pankajmathur/orca_minis_uncensored_dataset", "flozi00/conversations", "InfImagine/FakeImageDataset", "wyzelabs/RuleRecommendation", "squarelike/sharegpt_deepl_ko_translation", "gpt4life/alpaca_claud_filtered", "pankajmathur/orca_mini_v1_dataset", "nampdn-ai/tiny-bridgedict", "cmcjas/SDXL_ComfyUI_workflows", "rombodawg/MegaCodeTraining",
    "morpheuslord/cve-llm-training", "ymoslem/Law-StackExchange", "krisfu/awesome-llm-datasets-only-Chinese", "TaylorAI/pubmed_commercial", "kyujinpy/KoCoT_2000", "mychen76/invoices-and-receipts_ocr_v1", "kunishou/amenokaku-code-instruct", "approximatelabs/tablib-v1-sample", "swj0419/WikiMIA", "llmware/rag_instruct_test_dataset_0.1",
    "rizerphe/glaive-function-calling-v2-zephyr", "yuyijiong/Book_Summary_Chinese", "winglian/no_robots_rlhf", "castorini/wura", "diffusers/benchmarks", "nuprl/EditPackFT", "craigwu/vstar_bench", "Undi95/toxic-dpo-v0.1-sharegpt", "kunishou/oasst2-135k-ja", "ChuckMcSneed/WolframRavenwolfs_benchmark_results",
    "CausalLM/GPT-4-Self-Instruct-Japanese", "jtatman/stable-diffusion-prompts-uncensored", "lowres/anime", "MediaTek-Research/TCEval-v2", "AGBonnet/augmented-clinical-notes", "HuggingFaceH4/cai-conversation-harmless", "lmms-lab/VQAv2", "lmms-lab/DocVQA", "Mutonix/RefGPT-Fact-v2", "ba188/NHS_HES",
    "ajibawa-2023/Children-Stories-Collection", "Vikhrmodels/LLaVA-Instruct-ru", "Doctor-Shotgun/theory-of-mind-dpo", "divyasharma0795/AppleVisionPro_Tweets", "TIGER-Lab/MATH-plus", "cgato/SlimOrcaDedupCleaned", "YanweiLi/MGM-Pretrain", "HuggingFaceH4/llava-instruct-mix-vsft", "fal-ai/imgsys-results", "mzbac/function-calling-llama-3-format-v1.1",
    "Yale-LILY/aeslc", "google-research-datasets/aquamuse", "allenai/atomic", "CFPB/consumer-finance-complaints", "rishitdagli/cppe-5", "stanfordnlp/craigslist_bargains", "fquad", "google_wellformed_query", "interpress_news_category_tr_lite", "thu-coai/kd_conv_with_kb",
    "kakaobrain/kor_nli", "ParaPat/para_pat", "google-research-datasets/poem_sentiment", "eusip/silicone", "LSDSem/story_cloze", "turkic-interlingua/turkic_xwmt", "bea2019st/wi_locness", "fancyzhx/yelp_polarity", "CodedotAI/code_clippy", "SetFit/sst5",
    "deepset/germandpr", "flax-sentence-embeddings/stackexchange_titlebody_best_and_down_voted_answer_jsonl", "microsoft/codexglue_method_generation", "nickmuchi/financial-classification", "uitnlp/vietnamese_students_feedback", "ydshieh/coco_dataset_script", "cgarciae/cartoonset", "DMetaSoul/chinese-semantic-textual-similarity", "ukr-models/Ukr-Synth", "Matthijs/snacks",
    "csebuetnlp/CrossSum", "Moo/korean-parallel-corpora", "HuggingFaceM4/TGIF", "khalidalt/tydiqa-goldp", "mteb/amazon_reviews_multi", "silver/mmchat", "fmplaza/offendes", "ColumbiaNLP/FLUTE", "tner/ontonotes5", "jordanparker6/publaynet",
    "tarteel-ai/quranqa", "OATML-Markslab/ProteinGym", "google/cvss", "RUCAIBox/Open-Dialogue", "cardiffnlp/tweet_topic_multi", "priyank-m/chinese_text_recognition", "skytnt/fbanimehq", "huggingface-projects/color-palettes-sd", "heegyu/namuwiki", "FremyCompany/BioLORD-Dataset",
    "nikitam/ACES", "nitrosocke/arcane-diffusion-dataset", "Twitter/TwitterFaveGraph", "ju-resplande/qa-pt", "Short-Answer-Feedback/saf_communication_networks_english", "hoskinson-center/proofnet", "Erythrocyte/Diff-SVC_Genshin_Datasets", "nyanko7/pixiv_top50", "ashraf-ali/quran-data", "Nerfgun3/splash_art",
    "nelorth/oxford-flowers", "laion/laion2b-en-vit-l-14-embeddings", "lsy641/PsyQA", "masakhane/masakhaner2", "alexandreteles/mental-health-conversational-data", "joelniklaus/legal_case_document_summarization", "Cohere/wikipedia-22-12-zh-embeddings", "ruanchaves/hatebr", "liyucheng/chinese_metaphor_dataset", "pierreguillou/DocLayNet-large",
    "range3/cc100-ja", "Supermaxman/esa-hubble", "Den4ikAI/russian_instructions_2", "nlpcloud/instructions-dataset-adapted-from-stanford-alpaca-for-gpt-j", "medalpaca/medical_meadow_mediqa", "InstaDeepAI/multi_species_genomes", "larryvrh/WikiMatrix-v1-Ja_Zh-filtered", "IlyaGusev/ru_sharegpt_cleaned", "LEAP/ClimSim_high-res", "niizam/4chan-datasets",
    "kunishou/databricks-dolly-69k-ja-en-translation", "enryu43/twitter100m_tweets", "heegyu/korquad-chat-v1", "griffin/ChemSum", "KakologArchives/KakologArchives", "openllmplayground/pandagpt_visual_instruction_dataset", "fujiki/japanese_alpaca_data", "zhiqings/dromedary-65b-verbose-clone-v0", "hammer888/interior_style_dataset", "edarchimbaud/timeseries-1m-stocks",
    "FremyCompany/AGCT-Dataset", "project-sloth/captcha-images", "jondurbin/rosettacode-raw", "collabora/whisperspeech", "microsoft/LCC_csharp", "YeungNLP/school_math_0.25M", "portuguese-benchmark-datasets/BLUEX", "globis-university/aozorabunko-clean", "totally-not-an-llm/sharegpt-hyperfiltered-3k", "DAMO-NLP-MT/multialpaca",
    "crumb/Wizard-EvolInstruct70k-k4", "d0rj/OpenOrca-ru", "jed351/Traditional-Chinese-Common-Crawl-Filtered", "v2ray/jannie-log", "abacusai/WikiQA-Altered_Numeric_QA", "ChrisHayduk/Llama-2-SQL-Dataset", "TempoFunk/hdvila-100M", "tyang816/MedChatZH", "Falah/image_generation_prompts_SDXL", "turing-motors/LLaVA-Instruct-150K-JA",
    "OpenAssistant/OASST-DE", "jitx/Methods2Test_java_unit_test_code", "llvm-ml/ComPile", "BleachNick/MIC_full", "bugdaryan/sql-create-context-instruction", "harvard-lil/cold-cases", "knowrohit07/ArithmeLogic", "mikonvergence/LAION-EO", "euclaise/writingprompts", "erhwenkuo/medical_dialogue-chinese-zhtw",
    "Nexusflow/NexusRaven_API_evaluation", "jackhhao/jailbreak-classification", "cmalaviya/expertqa", "meta-math/GSM8K_Backward", "jamescalam/ai-arxiv", "yuyijiong/Long-instruction-en2zh", "microsoft/kitab", "MemGPT/MSC-Self-Instruct", "AI-Secure/DecodingTrust", "ShashiVish/cover-letter-dataset",
    "umarigan/turkiye_finance_qa", "allenai/scirepeval", "tahrirchi/uz-books", "yuyijiong/LongPaper_multitask", "pseudolab/MedSi", "lavita/medical-qa-datasets", "vilm/OpenOrca-Viet", "kyujinpy/KOR-OpenOrca-Platypus-v3", "akemiH/NoteChat", "openerotica/erotiquant",
    "listen2you002/ChartLlama-Dataset", "saillab/taco-datasets", "nuprl/CanItEdit", "kyujinpy/orca_math_dpo", "adamkarvonen/chess_games", "blancsw/oasst2_top1_chat_format", "Awiny/Howto-Interlink7M", "NobodyExistsOnTheInternet/ToxicDPOqa", "VatsaDev/worldbuild", "lorinma/NL2SQL_zh",
    "mlabonne/chessllm", "genggui001/gg_zh_v1_550B", "DL3DV/DL3DV-ALL-4K", "paraloq/json_data_extraction", "tastypear/unalignment-toxic-dpo-v0.2-zh_cn", "hpprc/jawiki", "eduagarcia/LegalPT_dedup", "christopherthompson81/quant_exploration", "alvarobartt/dpo-mix-7k-simplified", "ucekmez/OpenOrca-tr",
    "ehristoforu/dalle-3-images", "ivrit-ai/whisper-training", "SPRIGHT-T2I/spright", "coseal/CodeUltraFeedback_binarized", "ParasiticRogue/Bluemoon-Light", "wdndev/webnovel-chinese", "jondurbin/bagel-v0.5", "Lin-Chen/MMStar", "tolgadev/turkish_73k_instruct_extended", "Babelscape/ALERT_DPO",
    "kigner/ruozhiba-llama3", "davanstrien/dataset-tldr-preference-dpo", "facebook/asset", "barilan/blog_authorship_corpus", "dataset-org/c3", "clinc/clinc_oos", "eli5_category", "mohnish/lc_quad", "lm1b", "ParaCrawl/para_crawl",
    "crscardellino/spanish_billion_words", "KorQuAD/squad_kor_v2", "nunorc/squad_v1_pt", "cgpotts/swda", "nakhun/thaisum", "wmt/wmt14", "SetFit/20_newsgroups", "bertin-project/mc4-sampling", "lbox/lbox_open", "codeparrot/codeparrot-clean-train",
    "thomwolf/github-python", "Adapting/empathetic_dialogues_v2", "Bingsu/Human_Action_Recognition", "mustapha/QuranExe", "ceyda/fashion-products-small", "frgfm/imagenette", "naver-clova-ix/synthdog-en", "bigscience/evaluation-results", "pcuenq/oxford-pets", "SLPL/syntran-fa",
    "RUCAIBox/Story-Generation", "jonathanli/law-stack-exchange", "ai-forever/school_notebooks_RU", "ashraq/esc50", "waifu-research-department/regularization", "sbx/superlim-2", "ashraq/financial-news", "AluminiumOxide/personal_latent_diffusion", "elenanereiss/german-ler", "Nerfgun3/flower_style",
    "lmqg/qa_harvesting_from_wikipedia", "Nerfgun3/land_style", "NeelNanda/counterfact-tracing", "VietAI/vi_pubmed", "andyyang/stable_diffusion_prompts_2m", "its5Q/yandex-q", "wanng/laion-high-resolution-chinese", "Salesforce/rose", "Jean-Baptiste/financial_news_sentiment", "diltdicker/romance_novel_data-2022",
}
other_datasets = {"OpenCo7/UpVoteWeb"}
enabled_datasets = top_2k_most_liked_datasets | other_datasets
