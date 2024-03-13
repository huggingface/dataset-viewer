import logging
import re
from collections.abc import Mapping
from http import HTTPStatus
from itertools import islice
from typing import Any, Optional, Union

from datasets import ClassLabel, Features, Image, Value
from libapi.authentication import auth_check
from libapi.exceptions import (
    ApiError,
    MissingRequiredParameterError,
    UnexpectedApiError,
)
from libapi.request import get_request_parameter
from libapi.utils import (
    Endpoint,
    are_valid_parameters,
    get_cache_entry_from_step,
    get_json_api_error_response,
    get_json_error_response,
    get_json_ok_response,
)
from libcommon.constants import CROISSANT_MAX_CONFIGS, DATASET_INFO_KIND
from libcommon.croissant_utils import get_record_set
from libcommon.prometheus import StepProfiler
from libcommon.storage_client import StorageClient
from starlette.requests import Request
from starlette.responses import Response

MAX_COLUMNS = 1_000
# ^ same value as the default for FIRST_ROWS_COLUMNS_MAX_NUMBER (see services/worker)


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
    """Escapes names and IDs in Croissant.

    Reasons:
    - `/` are used in the syntax as delimiters. So we replace them.
    - Two FileObject/FileSet/RecordSet/Fields cannot have the same ID. So we append a postfix in case it happens.

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


def get_croissant_from_dataset_infos(
    dataset: str, infos: list[Mapping[str, Any]], partial: bool, full_jsonld: bool, is_v1: bool
) -> Mapping[str, Any]:
    repo_name = "repo"
    names: set[str] = set(repo_name)
    distribution = [
        _remove_none_values(
            {
                "@type": "sc:FileObject",
                "@id": repo_name if is_v1 else None,
                "name": repo_name,
                "description": "The Hugging Face git repository.",
                "contentUrl": f"https://huggingface.co/datasets/{dataset}/tree/refs%2Fconvert%2Fparquet",
                "encodingFormat": "git+https",
                "sha256": "https://github.com/mlcommons/croissant/issues/80",
            }
        )
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
            _remove_none_values(
                {
                    "@type": "sc:FileSet",
                    "@id": distribution_name if is_v1 else None,
                    "name": distribution_name,
                    "containedIn": {"@id": repo_name} if is_v1 else repo_name,
                    "encodingFormat": "application/x-parquet",
                    "includes": f"{config}/*/*.parquet",
                }
            )
        )
        skipped_columns = []
        for column, feature in features.items():
            if len(fields) >= MAX_COLUMNS and not full_jsonld:
                description_body += f"\n- {len(features) - MAX_COLUMNS} skipped column{'s' if len(features) - MAX_COLUMNS > 1 else ''} (max number of columns reached)"
                break
            fields_names: set[str] = set()
            if isinstance(feature, Value) and feature.dtype in HF_TO_CROISSANT_VALUE_TYPE:
                if is_v1:
                    field_source = {"fileSet": {"@id": distribution_name}, "extract": {"column": column}}
                else:
                    field_source = {"distribution": distribution_name, "extract": {"column": column}}
                field_name = _escape_name(column, fields_names)
                fields.append(
                    _remove_none_values(
                        {
                            "@type": "cr:Field" if is_v1 else "ml:Field",
                            "@id": field_name if is_v1 else None,
                            "name": field_name,
                            "description": f"Column '{column}' from the Hugging Face parquet file.",
                            "dataType": HF_TO_CROISSANT_VALUE_TYPE[feature.dtype],
                            "source": field_source,
                        }
                    )
                )
            elif isinstance(feature, Image):
                field_name = _escape_name(column, fields_names)
                if is_v1:
                    field_source = {
                        "fileSet": {"@id": distribution_name},
                        "extract": {"column": column},
                        "transform": {"jsonPath": "bytes"},
                    }
                else:
                    field_source = {
                            "distribution": distribution_name,
                            "extract": {"column": column},
                            "transform": {"jsonPath": "bytes"},
                    }
                fields.append(
                    _remove_none_values(
                        {
                            "@type": "cr:Field" if is_v1 else "ml:Field",
                            "@id": field_name if is_v1 else None,
                            "name": field_name,
                            "description": f"Image column '{column}' from the Hugging Face parquet file.",
                            "dataType": "sc:ImageObject",
                            "source": field_source,
                        }
                    )
                )
            elif isinstance(feature, ClassLabel):
                field_name = _escape_name(column, fields_names)
                if is_v1:
                    field_source = {"fileSet": {"@id": distribution_name}, "extract": {"column": column}}
                else:
                    field_source = {"distribution": distribution_name, "extract": {"column": column}}
                fields.append(
                    _remove_none_values(
                        {
                            "@type": "cr:Field" if is_v1 else "ml:Field",
                            "@id": field_name if is_v1 else None,
                            "name": field_name,
                            "description": f"ClassLabel column '{column}' from the Hugging Face parquet file.\nLabels:\n"
                            + ", ".join(f"{name} ({i})" for i, name in enumerate(feature.names)),
                            "dataType": "sc:Integer",
                            "source": field_source,
                        }
                    )
                )
            else:
                skipped_columns.append(column)
        record_set_name = get_record_set(dataset=dataset, config_name=config)
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
        record_set_name = _escape_name(record_set_name, names)
        record_set.append(
            _remove_none_values(
                {
                    "@type": "cr:RecordSet" if is_v1 else "ml:RecordSet",
                    "@id": record_set_name if is_v1 else None,
                    "name": record_set_name,
                    "description": description,
                    "field": fields,
                }
            )
        )
    prefix = "ml" if not is_v1 else "cr"
    context = _remove_none_values(
        {
                "@language": "en",
                "@vocab": "https://schema.org/",
                "citeAs": "cr:citeAs" if is_v1 else None,
                "column": f"{prefix}:column",
                "conformsTo": "dct:conformsTo" if is_v1 else None,
                "cr": "http://mlcommons.org/croissant/" if is_v1 else None,
                "data": {
                    "@id": f"{prefix}:data",
                    "@type": "@json"
                },
                "dataBiases": f"{prefix}:dataBiases",
                "dataCollection": f"{prefix}:dataCollection",
                "dataType": {
                    "@id": f"{prefix}:dataType",
                    "@type": "@vocab"
                },
                "dct": "http://purl.org/dc/terms/",
                "extract": f"{prefix}:extract",
                "field": f"{prefix}:field",
                "fileProperty": f"{prefix}:fileProperty",
                "fileObject": "cr:fileObject" if is_v1 else None,
                "fileSet": "cr:fileSet" if is_v1 else None,
                "format": f"{prefix}:format",
                "includes": f"{prefix}:includes",
                "isEnumeration": f"{prefix}:isEnumeration",
                "isLiveDataset": "cr:isLiveDataset" if is_v1 else None,
                "jsonPath": f"{prefix}:jsonPath",
                "ml": None if is_v1 else "http://mlcommons.org/schema/",
                "key": "cr:key" if is_v1 else None,
                "md5": "cr:md5" if is_v1 else None,
                "parentField": f"{prefix}:parentField",
                "path": f"{prefix}:path",
                "personalSensitiveInformation": f"{prefix}:personalSensitiveInformation",
                "recordSet": f"{prefix}:recordSet",
                "references": f"{prefix}:references",
                "regex": f"{prefix}:regex",
                "repeated": f"{prefix}:repeated",
                "replace": f"{prefix}:replace",
                "sc": "https://schema.org/",
                "separator": f"{prefix}:separator",
                "source": f"{prefix}:source",
                "subField": f"{prefix}:subField",
                "transform": f"{prefix}:transform",
            }
      )
    return _remove_none_values(
        {
            "@context": context,
            "@type": "sc:Dataset",
            "name": _escape_name(dataset, names),
            "conformsTo": "http://mlcommons.org/croissant/1.0" if is_v1 else None,
            "description": f"{dataset} dataset hosted on Hugging Face and contributed by the HF Datasets community",
            "identifier": identifier,
            "license": _license,
            "url": f"https://huggingface.co/datasets/{dataset}",
            "distribution": distribution,
            "recordSet": record_set,
        }
    )


def _get_full_jsonld_parameter(request: Request) -> bool:
    """Parameter to retrieve the full JSON-LD (full=True) or a truncated/abridged JSON-LD (full=False) with less features."""
    full_jsonld = get_request_parameter(request, "full", default="true")
    if full_jsonld.lower() == "false":
        return False
    return True


def _get_is_v1(request: Request) -> bool:
    """Whether the output follows Croissant 1.0 or 0.8. Defaults to True."""
    isV1 = get_request_parameter(request, "isV1", default="true")
    if isV1.lower() == "false":
        return False
    return True


def create_croissant_endpoint(
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    hf_jwt_public_keys: Optional[list[str]] = None,
    hf_jwt_algorithm: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
    storage_clients: Optional[list[StorageClient]] = None,
) -> Endpoint:
    async def croissant_endpoint(request: Request) -> Response:
        endpoint_name = "croissant"
        context = f"endpoint: {endpoint_name}"
        revision: Optional[str] = None
        with StepProfiler(method="croissant_endpoint", step="all", context=context):
            try:
                with StepProfiler(
                    method="croissant_endpoint",
                    step="validate parameters and get processing steps",
                    context=context,
                ):
                    full_jsonld = _get_full_jsonld_parameter(request)
                    isV1 = _get_is_v1(request)
                    dataset = get_request_parameter(request, "dataset")
                    logging.debug(f"endpoint={endpoint_name} dataset={dataset}")
                    if not are_valid_parameters([dataset]):
                        raise MissingRequiredParameterError("Parameter 'dataset' is required")
                # if auth_check fails, it will raise an exception that will be caught below
                with StepProfiler(method="croissant_endpoint", step="check authentication", context=context):
                    await auth_check(
                        dataset,
                        external_auth_url=external_auth_url,
                        request=request,
                        hf_jwt_public_keys=hf_jwt_public_keys,
                        hf_jwt_algorithm=hf_jwt_algorithm,
                        hf_timeout_seconds=hf_timeout_seconds,
                    )
                # getting result based on processing steps
                with StepProfiler(method="croissant_endpoint", step="get info cache entry", context=context):
                    info_result = get_cache_entry_from_step(
                        processing_step_name=DATASET_INFO_KIND,
                        dataset=dataset,
                        config=None,
                        split=None,
                        hf_endpoint=hf_endpoint,
                        hf_token=hf_token,
                        blocked_datasets=blocked_datasets,
                        hf_timeout_seconds=hf_timeout_seconds,
                        storage_clients=storage_clients,
                    )
                content = info_result["content"]
                http_status = info_result["http_status"]
                error_code = info_result["error_code"]
                revision = info_result["dataset_git_revision"]
                if http_status == HTTPStatus.OK:
                    infos = list(islice(content["dataset_info"].values(), CROISSANT_MAX_CONFIGS))
                    partial = content["partial"]
                    with StepProfiler(method="croissant_endpoint", step="generate croissant json", context=context):
                        croissant = get_croissant_from_dataset_infos(
                            dataset=dataset,
                            infos=infos,
                            partial=partial,
                            full_jsonld=full_jsonld,
                            is_v1=isV1,
                        )
                    with StepProfiler(method="croissant_endpoint", step="generate OK response", context=context):
                        return get_json_ok_response(content=croissant, max_age=max_age_long, revision=revision)
                else:
                    with StepProfiler(method="croissant_endpoint", step="generate error response", context=context):
                        return get_json_error_response(
                            content=content,
                            status_code=http_status,
                            max_age=max_age_short,
                            error_code=error_code,
                            revision=revision,
                        )
            except Exception as e:
                error = e if isinstance(e, ApiError) else UnexpectedApiError("Unexpected error.", e)
                with StepProfiler(method="croissant_endpoint", step="generate API error response", context=context):
                    return get_json_api_error_response(error=error, max_age=max_age_short, revision=revision)

    return croissant_endpoint
