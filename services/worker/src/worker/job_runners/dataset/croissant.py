# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import re
from collections.abc import Mapping
from itertools import islice
from typing import Any, Union

from datasets import ClassLabel, Features, Image, Value
from libcommon.constants import CROISSANT_MAX_CONFIGS
from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import (
    get_previous_step_or_raise,
)

from worker.dtos import (
    CompleteJobResult,
    DatasetCroissantResponse,
)
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner

HF_TO_CROISSANT_VALUE_TYPE = {
    "string": "sc:Text",
    "int32": "sc:Integer",
    "int64": "sc:Integer",
    "float32": "sc:Float",
    "float64": "sc:Float",
    "bool": "sc:Boolean",
}

NAME_PATTERN_REGEX = "[^a-zA-Z0-9\\-_\\.]"


def _escape_name(name: str, names: set[str]) -> str:
    """Escapes names in Croissant.

    Reasons:
    - `/` are used in the syntax as delimiters. So we replace them.
    - Two FileObject/FileSet/RecordSet/Fields cannot have the same name. So we append a postfix in case it happens.

    Args:
        name: The initial non-escaped name.
        names: The set of already existing names.
    Returns:
        `str`: The escaped name.
    """
    escaped_name = re.sub(NAME_PATTERN_REGEX, "_", name)
    while escaped_name in names:
        escaped_name = f"{escaped_name}_0"
    names.add(escaped_name)
    return escaped_name


def _extract_doi_tag(info: Mapping[str, Any]) -> Union[str, None]:
    """Extracts https://huggingface.co/docs/hub/en/doi."""
    tags = info.get("tags", [])
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("doi:"):
                return tag.replace("doi:", "", 1)
    return None


def _remove_none_values(json: Mapping[str, Any]) -> Mapping[str, Any]:
    """Removes None values in the first depth of a dict."""
    return {k: v for k, v in json.items() if v is not None}


def get_croissant_from_dataset_infos(dataset: str, infos: list[Mapping[str, Any]], partial: bool) -> Mapping[str, Any]:
    repo_name = "repo"
    names: set[str] = set(repo_name)
    distribution = [
        {
            "@type": "sc:FileObject",
            "name": repo_name,
            "description": "The Hugging Face git repository.",
            "contentUrl": f"https://huggingface.co/datasets/{dataset}/tree/refs%2Fconvert%2Fparquet",
            "encodingFormat": "git+https",
            "sha256": "https://github.com/mlcommons/croissant/issues/80",
        }
    ]
    identifier = None
    _license = None
    record_set = []
    for info in infos:
        description_body = ""
        _license = info.get("license")
        identifier = _extract_doi_tag(info)
        config = info["config_name"]
        features = Features.from_dict(info["features"])
        fields: list[dict[str, Any]] = []
        splits = list(info["splits"])
        distribution_name = _escape_name(f"parquet-files-for-config-{config}", names)
        distribution.append(
            {
                "@type": "sc:FileSet",
                "name": distribution_name,
                "containedIn": repo_name,
                "encodingFormat": "application/x-parquet",
                "includes": f"{config}/*/*.parquet",
            }
        )
        skipped_columns = []
        for column, feature in features.items():
            fields_names: set[str] = set()
            if isinstance(feature, Value) and feature.dtype in HF_TO_CROISSANT_VALUE_TYPE:
                fields.append(
                    {
                        "@type": "ml:Field",
                        "name": _escape_name(column, fields_names),
                        "description": f"Column '{column}' from the Hugging Face parquet file.",
                        "dataType": HF_TO_CROISSANT_VALUE_TYPE[feature.dtype],
                        "source": {"distribution": distribution_name, "extract": {"column": column}},
                    }
                )
            elif isinstance(feature, Image):
                fields.append(
                    {
                        "@type": "ml:Field",
                        "name": _escape_name(column, fields_names),
                        "description": f"Image column '{column}' from the Hugging Face parquet file.",
                        "dataType": "sc:ImageObject",
                        "source": {
                            "distribution": distribution_name,
                            "extract": {"column": column},
                            "transform": {"jsonPath": "bytes"},
                        },
                    }
                )
            elif isinstance(feature, ClassLabel):
                fields.append(
                    {
                        "@type": "ml:Field",
                        "name": _escape_name(column, fields_names),
                        "description": f"ClassLabel column '{column}' from the Hugging Face parquet file.\nLabels:\n"
                        + ", ".join(f"{name} ({i})" for i, name in enumerate(feature.names)),
                        "dataType": "sc:Integer",
                        "source": {"distribution": distribution_name, "extract": {"column": column}},
                    }
                )
            else:
                skipped_columns.append(column)
        record_set_name = config if config != dataset else f"record_set_{config}"
        description = f"{dataset} - '{config}' subset"
        if partial:
            description += " (first 5GB)"
        if len(splits) > 1:
            description_body += f"\n- {len(splits)} split{'s' if len(splits) > 1 else ''}: {', '.join(splits)}"
        if skipped_columns:
            description_body += f"\n- {len(skipped_columns)} skipped column{'s' if len(skipped_columns) > 1 else ''}: {', '.join(skipped_columns)}"
        if description_body:
            description += "\n\nAdditional information:"
            description += description_body
        record_set.append(
            {
                "@type": "ml:RecordSet",
                "name": _escape_name(record_set_name, names),
                "description": description,
                "field": fields,
            }
        )
    return _remove_none_values(
        {
            "@context": {
                "@language": "en",
                "@vocab": "https://schema.org/",
                "column": "ml:column",
                "data": {
                    "@id": "ml:data",
                    "@type": "@json",
                },
                "dataType": {
                    "@id": "ml:dataType",
                    "@type": "@vocab",
                },
                "extract": "ml:extract",
                "field": "ml:field",
                "fileProperty": "ml:fileProperty",
                "format": "ml:format",
                "includes": "ml:includes",
                "isEnumeration": "ml:isEnumeration",
                "jsonPath": "ml:jsonPath",
                "ml": "http://mlcommons.org/schema/",
                "parentField": "ml:parentField",
                "path": "ml:path",
                "recordSet": "ml:recordSet",
                "references": "ml:references",
                "regex": "ml:regex",
                "repeated": "ml:repeated",
                "replace": "ml:replace",
                "sc": "https://schema.org/",
                "separator": "ml:separator",
                "source": "ml:source",
                "subField": "ml:subField",
                "transform": "ml:transform",
            },
            "@type": "sc:Dataset",
            "name": _escape_name(dataset, names),
            "description": f"{dataset} dataset hosted on Hugging Face and contributed by the HF Datasets community",
            "identifier": identifier,
            "license": _license,
            "url": f"https://huggingface.co/datasets/{dataset}",
            "distribution": distribution,
            "recordSet": record_set,
        }
    )


def compute_croissant_response(dataset: str) -> DatasetCroissantResponse:
    """
    Get the response of 'dataset-croissant' for one specific dataset on huggingface.co.

    If the dataset contains more than 100 configs, only the first 100 configs are included in the croissant metadata.

    Here, we don't truncate the number of fields. See the /croissant endpoint implementation fo truncation of the fields.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.

    Raises:
        [~`libcommon.simple_cache.CachedArtifactError`]:
          If the previous step gave an error.
        [~`libcommon.exceptions.PreviousStepFormatError`]:
            If the content of the previous step has not the expected format

    Returns:
        `DatasetCroissantResponse`: The croissant response (schema metadata). It has fields:
        - **croissant** (`dict[str, Any]`): the croissant metadata.
        - **truncated_configs** (`bool`): true if only the first 100 configs are included in the croissant metadata,
          but the dataset had more than 100 configs.
        - **partial** (`bool`): true if the dataset is partial (see partial conversion to Parquet).
    """
    logging.info(f"compute 'dataset-croissant' for {dataset=}")

    dataset_info_response = get_previous_step_or_raise(kind="dataset-info", dataset=dataset)
    try:
        content = dataset_info_response["content"]
        truncated_configs = len(content["dataset_info"]) > CROISSANT_MAX_CONFIGS
        infos = list(islice(content["dataset_info"].values(), CROISSANT_MAX_CONFIGS))
        partial = content["partial"]
        croissant = get_croissant_from_dataset_infos(dataset=dataset, infos=infos, partial=partial)
    except KeyError as e:
        raise PreviousStepFormatError("Previous step 'dataset-info' did not return the expected content.", e) from e
    return DatasetCroissantResponse(croissant=croissant, truncated_configs=truncated_configs, partial=partial)


class DatasetCroissantJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-croissant"

    def compute(self) -> CompleteJobResult:
        response_content = compute_croissant_response(dataset=self.dataset)
        return CompleteJobResult(response_content)
