import logging
from collections.abc import Mapping
from http import HTTPStatus
from itertools import islice
from typing import Any, Optional, TypedDict

from datasets import Features, Image, Value
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
    get_cache_entry_from_steps,
    get_json_api_error_response,
    get_json_error_response,
    get_json_ok_response,
)
from libcommon.processing_graph import ProcessingGraph
from libcommon.prometheus import StepProfiler
from starlette.requests import Request
from starlette.responses import Response

MAX_CONFIGS = 100


class ParquetFile(TypedDict):
    dataset: str
    config: str
    split: str
    url: str
    filename: str
    size: int


HF_TO_CRROISSANT_VALUE_TYPE = {
    "string": "sc:Text",
    "int32": "sc:Integer",
    "int64": "sc:Integer",
    "float32": "sc:Float",
    "float64": "sc:Float",
    "bool": "sc:Boolean",
}


def get_croissant_from_dataset_infos(dataset: str, infos: list[Mapping[str, Any]], partial: bool) -> Mapping[str, Any]:
    distribution = [
        {
            "@type": "sc:FileObject",
            "name": "repo",
            "description": "The Hugging Face git repository.",
            "contentUrl": f"https://huggingface.co/datasets/{dataset}",
            "encodingFormat": "git+https",
            "sha256": "https://github.com/mlcommons/croissant/issues/80",
        }
    ]
    record_set = []
    for info in infos:
        config = info["config_name"]
        features = Features.from_dict(info["features"])
        fields: list[dict[str, Any]] = []
        distribution_name = f"parquet-files-for-config-{config}"
        distribution.append(
            {
                "@type": "sc:FileSet",
                "name": distribution_name,
                "containedIn": "repo",
                "encodingFormat": "application/x-parquet",
                "includes": f"{config}/*/*.parquet",
            }
        )
        record_set.append(
            {
                "@type": "ml:RecordSet",
                "name": config,
                "description": f"'{config}' subset{' (first 5GB)' if partial else ''}",
                "field": fields,
            }
        )
        skipped_columns = []
        for column, feature in features.items():
            if isinstance(feature, Value) and feature.dtype in HF_TO_CRROISSANT_VALUE_TYPE:
                fields.append(
                    {
                        "@type": "ml:Field",
                        "name": column,
                        "description": f"Column '{column}' from Hugging Face parquet file.",
                        "dataType": HF_TO_CRROISSANT_VALUE_TYPE[feature.dtype],
                        "source": {"distribution": distribution_name, "extract": {"column": column}},
                    }
                )
            elif isinstance(feature, Image):
                fields.append(
                    {
                        "@type": "ml:Field",
                        "name": column,
                        "description": f"Image column '{column}' from Hugging Face parquet file.",
                        "dataType": HF_TO_CRROISSANT_VALUE_TYPE[feature.dtype],
                        "source": {
                            "distribution": distribution_name,
                            "extract": {"column": column},
                            "transform": {"jsonPath": "bytes"},
                        },
                    }
                )
            else:
                skipped_columns.append(column)
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
        "name": dataset,
        "description": f"{dataset} dataset hosted on Hugging Face and contributed by the HF Datasets community",
        "url": f"https://huggingface.co/datasets/{dataset}",
        "distribution": distribution,
        "recordSet": record_set,
    }


def create_croissant_endpoint(
    processing_graph: ProcessingGraph,
    cache_max_days: int,
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    hf_jwt_public_keys: Optional[list[str]] = None,
    hf_jwt_algorithm: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
) -> Endpoint:
    async def croissant_endpoint(request: Request) -> Response:
        endpoint_name = "croissant"
        context = f"endpoint: {endpoint_name}"
        revision: Optional[str] = None
        processing_steps = processing_graph.get_dataset_info_processing_steps()
        with StepProfiler(method="croissant_endpoint", step="all", context=context):
            try:
                with StepProfiler(
                    method="croissant_endpoint",
                    step="validate parameters and get processing steps",
                    context=context,
                ):
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
                    info_result = get_cache_entry_from_steps(
                        processing_steps=processing_steps,
                        dataset=dataset,
                        config=None,
                        split=None,
                        processing_graph=processing_graph,
                        hf_endpoint=hf_endpoint,
                        hf_token=hf_token,
                        blocked_datasets=blocked_datasets,
                        hf_timeout_seconds=hf_timeout_seconds,
                        cache_max_days=cache_max_days,
                    )
                content = info_result["content"]
                http_status = info_result["http_status"]
                error_code = info_result["error_code"]
                revision = info_result["dataset_git_revision"]
                if http_status == HTTPStatus.OK:
                    dataset_infos = list(islice(content["dataset_infos"].values(), MAX_CONFIGS))
                    partial = content["partial"]
                    with StepProfiler(method="croissant_endpoint", step="generate croissant json", context=context):
                        croissant = get_croissant_from_dataset_infos(
                            dataset=dataset, infos=dataset_infos, partial=partial
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
