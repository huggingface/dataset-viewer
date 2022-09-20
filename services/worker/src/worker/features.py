# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Any

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
from numpy import ndarray  # type:ignore
from PIL import Image as PILImage  # type: ignore

from worker.asset import create_audio_files, create_image_file


def image(
    dataset: str,
    config: str,
    split: str,
    row_idx: int,
    value: Any,
    featureName: str,
    assets_base_url: str,
) -> Any:
    if value is None:
        return None
    if not isinstance(value, PILImage.Image):
        raise TypeError("image cell must be a PIL image")
    # attempt to generate one of the supported formats; if unsuccessful, throw an error
    for ext in [".jpg", ".png"]:
        try:
            return create_image_file(
                dataset, config, split, row_idx, featureName, f"image{ext}", value, assets_base_url
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
) -> Any:
    if value is None:
        return None
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
    return create_audio_files(dataset, config, split, row_idx, featureName, array, sampling_rate, assets_base_url)


# should we return both the value (as given by datasets) and the additional contents (audio files, image files)?
# in the case of the images or audio, if the value contains the raw data, it would take too much space and would
# trigger the response truncation -> less rows would be viewable
def get_cell_value(
    dataset: str,
    config: str,
    split: str,
    row_idx: int,
    cell: Any,
    featureName: str,
    fieldType: Any,
    assets_base_url: str,
) -> Any:
    if isinstance(fieldType, Image):
        return image(dataset, config, split, row_idx, cell, featureName, assets_base_url)
    elif isinstance(fieldType, Audio):
        return audio(dataset, config, split, row_idx, cell, featureName, assets_base_url)
    elif (
        isinstance(fieldType, Value)
        or isinstance(fieldType, ClassLabel)
        or isinstance(fieldType, Array2D)
        or isinstance(fieldType, Array3D)
        or isinstance(fieldType, Array4D)
        or isinstance(fieldType, Array5D)
        or isinstance(fieldType, Translation)
        or isinstance(fieldType, TranslationVariableLanguages)
        or isinstance(fieldType, Sequence)
        or isinstance(fieldType, list)
        or isinstance(fieldType, dict)
    ):
        return cell
    else:
        raise TypeError("could not determine the type of the data cell.")
