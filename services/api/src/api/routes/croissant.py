import logging
import re
from collections.abc import Mapping
from http import HTTPStatus
from itertools import islice
from typing import Any, Optional

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


def get_croissant_from_dataset_infos(
    dataset: str, infos: list[Mapping[str, Any]], partial: bool, full_jsonld: bool
) -> Mapping[str, Any]:
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
    record_set = []
    for info in infos:
        description_body = ""
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
            if len(fields) >= MAX_COLUMNS and not full_jsonld:
                description_body += f"\n- {len(features) - MAX_COLUMNS} skipped column{'s' if len(features) - MAX_COLUMNS > 1 else ''} (max number of columns reached)"
                break
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
    return {
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
        "url": f"https://huggingface.co/datasets/{dataset}",
        "distribution": distribution,
        "recordSet": record_set,
    }


def _get_full_jsonld_parameter(request: Request) -> bool:
    """Parameter to retrieve the full JSON-LD (full=True) or a truncated/abridged JSON-LD (full=False) with less features."""
    full_jsonld = get_request_parameter(request, "full", default="true")
    if full_jsonld.lower() == "false":
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
                            dataset=dataset, infos=infos, partial=partial, full_jsonld=full_jsonld
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
