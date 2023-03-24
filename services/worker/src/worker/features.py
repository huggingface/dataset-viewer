# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import io
import json
from typing import Any, List, Optional, Union
from zlib import adler32

from datasets import (
    Array2D,
    Array3D,
    Array4D,
    Array5D,
    Audio,
    ClassLabel,
    Image,
    Sequence,
    Translation,
    TranslationVariableLanguages,
    Value,
)
from libcommon.storage import StrPath
from numpy import ndarray
from PIL import Image as PILImage  # type: ignore

from worker.asset import (
    create_audio_files,
    create_audio_files_from_bytes,
    create_image_file,
)


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
) -> Any:
    if value is None:
        return None
    if not isinstance(value, PILImage.Image):
        try:
            image_bytes = value["bytes"]
            value = PILImage.open(io.BytesIO(image_bytes))
        except Exception:
            raise TypeError("image cell must be a PIL image")
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
) -> Any:
    if value is None:
        return None
    if "bytes" in value:
        return create_audio_files_from_bytes(
            dataset=dataset,
            config=config,
            split=split,
            row_idx=row_idx,
            column=featureName,
            array=value["bytes"],
            assets_base_url=assets_base_url,
            filename_base=append_hash_suffix("audio", json_path),
            assets_directory=assets_directory,
        )
    try:
        array = value["array"]
        sampling_rate = value["sampling_rate"]
    except Exception as e:
        raise TypeError("audio cell must contain 'array' and 'sampling_rate' fields") from e
    if type(array) != ndarray:
        raise TypeError("'array' field must be a numpy.ndarray")
    if type(sampling_rate) != int:
        raise TypeError("'sampling_rate' field must be an integer")
    # this function can raise, we don't catch it
    return create_audio_files(
        dataset=dataset,
        config=config,
        split=split,
        row_idx=row_idx,
        column=featureName,
        array=array,
        sampling_rate=sampling_rate,
        assets_base_url=assets_base_url,
        filename_base=append_hash_suffix("audio", json_path),
        assets_directory=assets_directory,
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
