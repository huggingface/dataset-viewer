# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from collections.abc import Mapping
from typing import Any, Optional, Union

from datasets import ClassLabel, Image, LargeList, Sequence, Value


def get_record_set(dataset: str, config_name: str) -> str:
    # Identical keys are not supported in Croissant
    # The current workaround that is used in /croissant endpoint
    # is to prefix the config name with `record_set_` if necessary.
    if dataset != config_name:
        return config_name
    else:
        return f"record_set_{config_name}"


MAX_COLUMNS = 1_000
# ^ same value as the default for FIRST_ROWS_COLUMNS_MAX_NUMBER (see services/worker)


def truncate_features_from_croissant_crumbs_response(content: Mapping[str, Any]) -> None:
    """Truncate the features from a croissant-crumbs response to avoid returning a large response."""
    if isinstance(content, dict) and "recordSet" in content and isinstance(content["recordSet"], list):
        for record in content["recordSet"]:
            if (
                isinstance(record, dict)
                and "field" in record
                and isinstance(record["field"], list)
                and len(record["field"]) > MAX_COLUMNS
            ):
                num_columns = len(record["field"])
                record["field"] = record["field"][:MAX_COLUMNS]
                record["description"] += (
                    f"\n- {num_columns - MAX_COLUMNS} skipped column{'s' if num_columns - MAX_COLUMNS > 1 else ''} (max number of columns reached)"
                )


HF_TO_CROISSANT_VALUE_TYPE = {
    "binary": "sc:Text",
    "bool": "sc:Boolean",
    "date32": "sc:Date",
    "date64": "sc:Date",
    "float8": "sc:Float",
    "float16": "sc:Float",
    "float32": "sc:Float",
    "float64": "sc:Float",
    "int8": "sc:Integer",
    "int16": "sc:Integer",
    "int32": "sc:Integer",
    "int64": "sc:Integer",
    "large_string": "sc:Text",
    "string": "sc:Text",
    "time32": "sc:Date",
    "time64": "sc:Date",
    "timestamp[ns]": "sc:Date",
    "uint8": "sc:Integer",
    "uint16": "sc:Integer",
    "uint32": "sc:Integer",
    "uint64": "sc:Integer",
}


def get_source(
    distribution_name: str, column: str, add_transform: bool, json_path: Optional[str] = None
) -> dict[str, Any]:
    """Returns a Source dictionary for a Field."""
    source = {"fileSet": {"@id": distribution_name}, "extract": {"column": column}}
    if add_transform and json_path:
        source["transform"] = {"jsonPath": json_path}
    return source


def feature_to_croissant_field(
    distribution_name: str,
    field_name: str,
    column: str,
    feature: Any,
    add_transform: bool = False,
    json_path: Optional[str] = None,
) -> Union[dict[str, Any], None]:
    """Converts a Hugging Face Datasets feature to a Croissant field or None if impossible."""
    if isinstance(feature, Value) and feature.dtype in HF_TO_CROISSANT_VALUE_TYPE:
        return {
            "@type": "cr:Field",
            "@id": field_name,
            "dataType": HF_TO_CROISSANT_VALUE_TYPE[feature.dtype],
            "source": get_source(distribution_name, column, add_transform, json_path),
        }
    elif isinstance(feature, Image):
        source = get_source(distribution_name, column, add_transform, json_path)
        if transform := source.get("transform"):
            source["transform"] = [transform, {"jsonPath": "bytes"}]
        else:
            source["transform"] = {"jsonPath": "bytes"}
        return {
            "@type": "cr:Field",
            "@id": field_name,
            "dataType": "sc:ImageObject",
            "source": source,
        }
    elif isinstance(feature, ClassLabel):
        return {
            "@type": "cr:Field",
            "@id": field_name,
            "dataType": "sc:Integer",
            "source": get_source(distribution_name, column, add_transform, json_path),
        }
    # Field with sub-fields.
    elif isinstance(feature, dict):
        return {
            "@type": "cr:Field",
            "@id": field_name,
            "subField": [
                feature_to_croissant_field(
                    distribution_name,
                    f"{field_name}/{subfeature_name}",
                    column,
                    sub_feature,
                    add_transform=True,
                    json_path=subfeature_name,
                )
                for subfeature_name, sub_feature in feature.items()
            ],
        }
    elif isinstance(feature, (Sequence, LargeList, list)):
        if isinstance(feature, (Sequence, LargeList)):
            sub_feature = feature.feature
        else:
            if len(feature) != 1:
                return None
            sub_feature = feature[0]
        field = feature_to_croissant_field(distribution_name, field_name, column, sub_feature)
        if field:
            field["repeated"] = True
            return field
    return None
