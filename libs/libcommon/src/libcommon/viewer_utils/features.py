# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import json
import os
from io import BytesIO
from tempfile import NamedTemporaryFile
from typing import Any, List, Optional, Tuple, Union
from zlib import adler32

import numpy as np
import soundfile  # type: ignore
from datasets import (
    Array2D,
    Array3D,
    Array4D,
    Array5D,
    Audio,
    ClassLabel,
    Features,
    Image,
    Sequence,
    Translation,
    TranslationVariableLanguages,
    Value,
)
from datasets.features.features import FeatureType, _visit
from PIL import Image as PILImage  # type: ignore

from libcommon.storage import StrPath
from libcommon.utils import FeatureItem
from libcommon.viewer_utils.asset import create_audio_file, create_image_file


def append_hash_suffix(string: str, json_path: Optional[List[Union[str, int]]] = None) -> str:
    """
    Hash the json path to a string.
    Args:
        string (``str``): The string to append the hash to.
        json_path (``list(str|int)``): the json path, which is a list of keys and indices
    Returns:
        the string suffixed with the hash of the json path

    Details:
    - no suffix if the list is empty
    - converted to hexadecimal to make the hash shorter
    - the 0x prefix is removed
    """
    return f"{string}-{hex(adler32(json.dumps(json_path).encode()))[2:]}" if json_path else string


def image(
    dataset: str,
    config: str,
    split: str,
    row_idx: int,
    value: Any,
    featureName: str,
    assets_base_url: str,
    assets_directory: StrPath,
    json_path: Optional[List[Union[str, int]]] = None,
    overwrite: bool = True,
) -> Any:
    if value is None:
        return None
    if isinstance(value, dict) and value.get("bytes"):
        value = PILImage.open(BytesIO(value["bytes"]))
    elif (
        isinstance(value, dict)
        and "path" in value
        and isinstance(value["path"], str)
        and os.path.exists(value["path"])
    ):
        value = PILImage.open(value["path"])
    if not isinstance(value, PILImage.Image):
        raise TypeError(
            "Image cell must be a PIL image or an encoded dict of an image, "
            f"but got {str(value)[:300]}{'...' if len(str(value)) > 300 else ''}"
        )
    # attempt to generate one of the supported formats; if unsuccessful, throw an error
    for ext in [".jpg", ".png"]:
        try:
            return create_image_file(
                dataset=dataset,
                config=config,
                split=split,
                row_idx=row_idx,
                column=featureName,
                filename=f"{append_hash_suffix('image', json_path)}{ext}",
                image=value,
                assets_base_url=assets_base_url,
                assets_directory=assets_directory,
                overwrite=overwrite,
            )
        except OSError:
            # if wrong format, try the next one, see https://github.com/huggingface/datasets-server/issues/191
            #  OSError: cannot write mode P as JPEG
            #  OSError: cannot write mode RGBA as JPEG
            continue
    raise ValueError("Image cannot be written as JPEG or PNG")


def audio(
    dataset: str,
    config: str,
    split: str,
    row_idx: int,
    value: Any,
    featureName: str,
    assets_base_url: str,
    assets_directory: StrPath,
    json_path: Optional[List[Union[str, int]]] = None,
    overwrite: bool = True,
) -> Any:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(
            "Audio cell must be an encoded dict of an audio sample, "
            f"but got {str(value)[:300]}{'...' if len(str(value)) > 300 else ''}"
        )
    if "path" in value and isinstance(value["path"], str):
        tmp_file_suffix = os.path.splitext(value["path"])[1]
    else:
        tmp_file_suffix = None
    with NamedTemporaryFile("wb", suffix=tmp_file_suffix) as tmp_audio_file:
        if "bytes" in value and isinstance(value["bytes"], bytes):
            with open(tmp_audio_file.name, "wb") as f:
                f.write(value["bytes"])
            audio_file_path = tmp_audio_file.name
        elif "path" in value and isinstance(value["path"], str) and os.path.exists(value["path"]):
            audio_file_path = value["path"]
        elif (
            "array" in value
            and isinstance(value["array"], np.ndarray)
            and "sampling_rate" in value
            and isinstance(value["sampling_rate"], int)
        ):
            soundfile.write(tmp_audio_file.name, value["array"], value["sampling_rate"], format="wav")
            audio_file_path = tmp_audio_file.name
        else:
            raise ValueError(f"An audio sample should have one of 'path' or 'bytes' but both are None in {value}.")
        # this function can raise, we don't catch it
        return create_audio_file(
            dataset=dataset,
            config=config,
            split=split,
            row_idx=row_idx,
            column=featureName,
            audio_file_path=audio_file_path,
            assets_base_url=assets_base_url,
            filename=f"{append_hash_suffix('audio', json_path)}.mp3",
            assets_directory=assets_directory,
            overwrite=overwrite,
        )


def get_cell_value(
    dataset: str,
    config: str,
    split: str,
    row_idx: int,
    cell: Any,
    featureName: str,
    fieldType: Any,
    assets_base_url: str,
    assets_directory: StrPath,
    json_path: Optional[List[Union[str, int]]] = None,
    overwrite: bool = True,
) -> Any:
    # always allow None values in the cells
    if cell is None:
        return cell
    if isinstance(fieldType, Image):
        return image(
            dataset=dataset,
            config=config,
            split=split,
            row_idx=row_idx,
            value=cell,
            featureName=featureName,
            assets_base_url=assets_base_url,
            assets_directory=assets_directory,
            json_path=json_path,
            overwrite=overwrite,
        )
    elif isinstance(fieldType, Audio):
        return audio(
            dataset=dataset,
            config=config,
            split=split,
            row_idx=row_idx,
            value=cell,
            featureName=featureName,
            assets_base_url=assets_base_url,
            assets_directory=assets_directory,
            json_path=json_path,
            overwrite=overwrite,
        )
    elif isinstance(fieldType, list):
        if type(cell) != list:
            raise TypeError("list cell must be a list.")
        if len(fieldType) != 1:
            raise TypeError("the feature type should be a 1-element list.")
        subFieldType = fieldType[0]
        return [
            get_cell_value(
                dataset=dataset,
                config=config,
                split=split,
                row_idx=row_idx,
                cell=subCell,
                featureName=featureName,
                fieldType=subFieldType,
                assets_base_url=assets_base_url,
                assets_directory=assets_directory,
                json_path=json_path + [idx] if json_path else [idx],
                overwrite=overwrite,
            )
            for (idx, subCell) in enumerate(cell)
        ]
    elif isinstance(fieldType, Sequence):
        if type(cell) == list:
            if fieldType.length >= 0 and len(cell) != fieldType.length:
                raise TypeError("the cell length should be the same as the Sequence length.")
            return [
                get_cell_value(
                    dataset=dataset,
                    config=config,
                    split=split,
                    row_idx=row_idx,
                    cell=subCell,
                    featureName=featureName,
                    fieldType=fieldType.feature,
                    assets_base_url=assets_base_url,
                    assets_directory=assets_directory,
                    json_path=json_path + [idx] if json_path else [idx],
                    overwrite=overwrite,
                )
                for (idx, subCell) in enumerate(cell)
            ]
        # if the internal feature of the Sequence is a dict, then the value will automatically
        # be converted into a dictionary of lists. See
        # https://huggingface.co/docs/datasets/v2.5.1/en/package_reference/main_classes#datasets.Features
        if type(cell) == dict:
            if any((type(v) != list) or (k not in fieldType.feature) for k, v in cell.items()):
                raise TypeError("The value of a Sequence of dicts should be a dictionary of lists.")
            return {
                key: [
                    get_cell_value(
                        dataset=dataset,
                        config=config,
                        split=split,
                        row_idx=row_idx,
                        cell=subCellItem,
                        featureName=featureName,
                        fieldType=fieldType.feature[key],
                        assets_base_url=assets_base_url,
                        assets_directory=assets_directory,
                        json_path=json_path + [key, idx] if json_path else [key, idx],
                        overwrite=overwrite,
                    )
                    for (idx, subCellItem) in enumerate(subCell)
                ]
                for (key, subCell) in cell.items()
            }
        raise TypeError("Sequence cell must be a list or a dict.")

    elif isinstance(fieldType, dict):
        if type(cell) != dict:
            raise TypeError("dict cell must be a dict.")
        return {
            key: get_cell_value(
                dataset=dataset,
                config=config,
                split=split,
                row_idx=row_idx,
                cell=subCell,
                featureName=featureName,
                fieldType=fieldType[key],
                assets_base_url=assets_base_url,
                assets_directory=assets_directory,
                json_path=json_path + [key] if json_path else [key],
                overwrite=overwrite,
            )
            for (key, subCell) in cell.items()
        }
    elif isinstance(
        fieldType,
        (
            Value,
            ClassLabel,
            Array2D,
            Array3D,
            Array4D,
            Array5D,
            Translation,
            TranslationVariableLanguages,
        ),
    ):
        return cell
    else:
        raise TypeError("could not determine the type of the data cell.")


# in JSON, dicts do not carry any order, so we need to return a list
#
# > An object is an *unordered* collection of zero or more name/value pairs, where a name is a string and a value
#   is a string, number, boolean, null, object, or array.
# > An array is an *ordered* sequence of zero or more values.
# > The terms "object" and "array" come from the conventions of JavaScript.
# from https://stackoverflow.com/a/7214312/7351594 / https://www.rfc-editor.org/rfc/rfc7159.html
def to_features_list(features: Features) -> List[FeatureItem]:
    features_dict = features.to_dict()
    return [
        {
            "feature_idx": idx,
            "name": name,
            "type": features_dict[name],
        }
        for idx, name in enumerate(features)
    ]


def get_supported_unsupported_columns(
    features: Features,
    unsupported_features: List[FeatureType] = [],
) -> Tuple[List[str], List[str]]:
    supported_columns, unsupported_columns = [], []

    for column, feature in features.items():
        str_column = str(column)
        supported = True

        def classify(feature: FeatureType) -> None:
            nonlocal supported
            for unsupported_feature in unsupported_features:
                if type(unsupported_feature) == type(feature) == Value:
                    if unsupported_feature.dtype == feature.dtype:
                        supported = False
                elif type(unsupported_feature) == type(feature):
                    supported = False

        _visit(feature, classify)
        if supported:
            supported_columns.append(str_column)
        else:
            unsupported_columns.append(str_column)
    return supported_columns, unsupported_columns
