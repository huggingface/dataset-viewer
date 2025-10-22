# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import logging
import re
from collections.abc import Mapping
from itertools import islice
from typing import Any

from datasets import Features
from libcommon.constants import CROISSANT_MAX_CONFIGS
from libcommon.croissant_utils import escape_ids, feature_to_croissant_field, get_record_set
from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import (
    get_previous_step_or_raise,
)

from worker.dtos import CompleteJobResult
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner


def _remove_none_values(json: Mapping[str, Any]) -> Mapping[str, Any]:
    """Removes None values in the first depth of a dict."""
    return {k: v for k, v in json.items() if v is not None}


def get_croissant_crumbs_from_dataset_infos(
    dataset: str, infos: list[Mapping[str, Any]], partial: bool, truncated_configs: bool
) -> Mapping[str, Any]:
    """Generates the "crumbs" of the Croissant JSON-LD metadata from the dataset infos.

    It's only a subset of the full JSON-LD metadata. See the Hugging Face API `/croissant` endpoint
    to get the complete Croissant JSON-LD metadata.
    """
    repo_name = "repo"
    ids: set[str] = set(repo_name)
    distribution = [
        _remove_none_values(
            {
                "@type": "cr:FileObject",
                "@id": repo_name,
                "name": repo_name,
                "description": "The Hugging Face git repository.",
                "contentUrl": f"https://huggingface.co/datasets/{dataset}/tree/refs%2Fconvert%2Fparquet",
                "encodingFormat": "git+https",
                "sha256": "https://github.com/mlcommons/croissant/issues/80",
            }
        )
    ]
    record_set = []
    
    # Check if dataset has geospatial modality
    is_geospatial = False
    try:
        dataset_modalities_response = get_previous_step_or_raise(kind="dataset-modalities", dataset=dataset)
        modalities = dataset_modalities_response["content"].get("modalities", [])
        is_geospatial = "geospatial" in modalities
    except Exception:
        # If modalities step fails, try direct file detection
        try:
            dataset_filetypes_response = get_previous_step_or_raise(kind="dataset-filetypes", dataset=dataset)
            content = dataset_filetypes_response["content"]
            if "filetypes" in content and isinstance(content["filetypes"], list):
                geospatial_extensions = {
                    ".shp", ".shx", ".dbf", ".prj", ".cpg", ".kml", ".kmz", ".gpx", 
                    ".geojson", ".topojson", ".gml", ".geoparquet", ".fgb",
                    ".img", ".bil", ".bip", ".bsq", ".gpkg", ".mbtiles", ".pmtiles",
                    ".tif", ".tiff"  # GeoTIFF files
                }
                for filetype in content["filetypes"]:
                    if filetype["extension"] in geospatial_extensions and filetype["count"] > 0:
                        is_geospatial = True
                        break
        except Exception:
            pass
    
    for info in infos:
        description_body = ""
        config = info["config_name"]
        features = Features.from_dict(info["features"])
        fields: list[dict[str, Any]] = []
        splits = list(info["splits"])
        distribution_name = escape_ids(f"parquet-files-for-config-{config}", ids)
        distribution.append(
            _remove_none_values(
                {
                    "@type": "cr:FileSet",
                    "@id": distribution_name,
                    "containedIn": {"@id": repo_name},
                    "encodingFormat": "application/x-parquet",
                    "includes": f"{config}/*/*.parquet",
                }
            )
        )
        skipped_columns = []
        record_set_name = get_record_set(dataset=dataset, config_name=config)
        record_set_name = escape_ids(record_set_name, ids)
        # Add splits record set.
        split_record_set_name = f"{record_set_name}_splits"
        split_field = _remove_none_values(
            {
                "@type": "cr:Field",
                "@id": f"{split_record_set_name}/split_name",
                "dataType": "sc:Text",
            }
        )
        record_set.append(
            _remove_none_values(
                {
                    "@type": "cr:RecordSet",
                    "dataType": "cr:Split",
                    "key": {"@id": f"{split_record_set_name}/split_name"},
                    "@id": split_record_set_name,
                    "name": split_record_set_name,
                    "description": f"Splits for the {record_set_name} config.",
                    "field": [split_field],
                    "data": [{f"{split_record_set_name}/split_name": split_name} for split_name in splits],
                }
            )
        )
        # Add a split field to the record set.
        piped_splits = "|".join([re.escape(split) for split in splits])
        fields.append(
            {
                "@type": "cr:Field",
                "@id": f"{record_set_name}/split",
                "dataType": "sc:Text",
                "source": {
                    "fileSet": {"@id": distribution_name},
                    "extract": {"fileProperty": "fullpath"},
                    "transform": {"regex": f"{re.escape(config)}/(?:partial-)?({piped_splits})/.+parquet$"},
                },
                "references": {"field": {"@id": f"{split_record_set_name}/split_name"}},
            }
        )
        for column, feature in features.items():
            fields_names: set[str] = set()
            field_name = f"{record_set_name}/{escape_ids(column, fields_names)}"
            field = feature_to_croissant_field(distribution_name, field_name, column, feature, ids)
            if field:
                fields.append(field)
            else:
                skipped_columns.append(column)
        description = f"{dataset} - '{config}' subset"
        if partial:
            description += " (first 5GB)"
        if truncated_configs:
            description += f" (only the first {CROISSANT_MAX_CONFIGS} subsets are included in this metadata)"
        if len(splits) > 1:
            description_body += f"\n- {len(splits)} split{'s' if len(splits) > 1 else ''}: {', '.join(splits)}"
        if skipped_columns:
            description_body += f"\n- {len(skipped_columns)} skipped column{'s' if len(skipped_columns) > 1 else ''}: {', '.join(skipped_columns)}"
        if description_body:
            description += "\n\nAdditional information:"
            description += description_body
        record_set.append(
            _remove_none_values(
                {
                    "@type": "cr:RecordSet",
                    "@id": record_set_name,
                    "description": description,
                    "field": fields,
                }
            )
        )
    context = {
        "@language": "en",
        "@vocab": "https://schema.org/",
        "arrayShape": "cr:arrayShape",
        "citeAs": "cr:citeAs",
        "column": "cr:column",
        "conformsTo": "dct:conformsTo",
        "cr": "http://mlcommons.org/croissant/",
        "data": {"@id": "cr:data", "@type": "@json"},
        "dataBiases": "cr:dataBiases",
        "dataCollection": "cr:dataCollection",
        "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
        "dct": "http://purl.org/dc/terms/",
        "extract": "cr:extract",
        "field": "cr:field",
        "fileProperty": "cr:fileProperty",
        "fileObject": "cr:fileObject",
        "fileSet": "cr:fileSet",
        "format": "cr:format",
        "includes": "cr:includes",
        "isArray": "cr:isArray",
        "isLiveDataset": "cr:isLiveDataset",
        "jsonPath": "cr:jsonPath",
        "key": "cr:key",
        "md5": "cr:md5",
        "parentField": "cr:parentField",
        "path": "cr:path",
        "personalSensitiveInformation": "cr:personalSensitiveInformation",
        "recordSet": "cr:recordSet",
        "references": "cr:references",
        "regex": "cr:regex",
        "repeated": "cr:repeated",
        "replace": "cr:replace",
        "sc": "https://schema.org/",
        "separator": "cr:separator",
        "source": "cr:source",
        "subField": "cr:subField",
        "transform": "cr:transform",
        # GeoCroissant properties
        "geocr": "http://mlcommons.org/croissant/geo/1.0",
        "boundingBox": "geocr:boundingBox",
        "geometry": "geocr:geometry",
        "resolution": "geocr:resolution",
        "crs": "geocr:crs",
        "temporalExtent": "geocr:temporalExtent",
        "spatialResolution": "geocr:spatialResolution",
        "temporalResolution": "geocr:temporalResolution",
        "label": "geocr:label",
        "image": "geocr:image",
    }
    # Prepare base output
    output = {
        "@context": context,
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "distribution": distribution,
        "recordSet": record_set,
    }
    
    # Add GeoCroissant properties if dataset is geospatial
    if is_geospatial:
        # TODO: Extract geospatial metadata from user-provided metadata.json or dataset card
        pass
    
    return _remove_none_values(output)


def compute_croissant_crumbs_response(dataset: str) -> Mapping[str, Any]:
    """
    Get the response of 'dataset-croissant-crumbs' for one specific dataset on huggingface.co.

    If the dataset contains more than 100 configs, only the first 100 configs are included in the croissant metadata parts.

    Here, we don't truncate the number of fields. See the /croissant-crumbs endpoint implementation in services/api for
      truncation of the fields.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.

    Raises:
        [~`libcommon.simple_cache.CachedArtifactError`]:
          If the previous step gave an error.
        [~`libcommon.exceptions.PreviousStepFormatError`]:
            If the content of the previous step has not the expected format

    Returns `dict[str, Any]`: the croissant metadata parts.
    """
    logging.info(f"compute 'dataset-croissant-crumbs' for {dataset=}")

    dataset_info_response = get_previous_step_or_raise(kind="dataset-info", dataset=dataset)
    try:
        content = dataset_info_response["content"]
        truncated_configs = len(content["dataset_info"]) > CROISSANT_MAX_CONFIGS
        infos = list(islice(content["dataset_info"].values(), CROISSANT_MAX_CONFIGS))
        partial = content["partial"]
        croissant_crumbs = get_croissant_crumbs_from_dataset_infos(
            dataset=dataset, infos=infos, partial=partial, truncated_configs=truncated_configs
        )
    except KeyError as e:
        raise PreviousStepFormatError("Previous step 'dataset-info' did not return the expected content.", e) from e
    return croissant_crumbs


class DatasetCroissantCrumbsJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-croissant-crumbs"

    def compute(self) -> CompleteJobResult:
        response_content = compute_croissant_crumbs_response(dataset=self.dataset)
        return CompleteJobResult(response_content)
