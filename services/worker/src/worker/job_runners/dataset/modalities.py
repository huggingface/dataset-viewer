# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus

from datasets import Audio, Features, Image, Sequence, Translation, TranslationVariableLanguages, Value
from datasets.features.features import FeatureType, _visit
from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import (
    CachedArtifactNotFoundError,
    get_previous_step_or_raise,
    get_response,
)

from worker.dtos import (
    CompleteJobResult,
    DatasetModalitiesResponse,
    DatasetModality,
)
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner


def detect_features_modalities(features: Features) -> set[DatasetModality]:
    """
    Detect modalities of a dataset using the features (column types).

    Args:
        features (`datasets.Features`):
            The features of a config.

    Returns:
        `set[DatasetModality]`: A set of modalities.
    """
    modalities: set[DatasetModality] = set()

    def classify_modality(feature: FeatureType) -> None:
        nonlocal modalities
        if isinstance(feature, Audio):
            modalities.add("audio")
        elif isinstance(feature, Image):
            modalities.add("image")
        elif isinstance(feature, Value) and feature.dtype in ("string", "large_string"):
            modalities.add("text")
        elif isinstance(feature, (Translation, TranslationVariableLanguages)):
            modalities.add("text")

    _visit(features, classify_modality)

    # detection of tabular data: if there are at least two top-level numerical columns, and no "media" columns
    if (
        not ("audio" in modalities or "image" in modalities)
        and len(
            [
                feature
                for feature in features.values()
                if isinstance(feature, Value) and ("int" in feature.dtype or "float" in feature.dtype)
            ]
        )
        >= 2
    ):
        modalities.add("tabular")

    # detection of time series
    if any(
        "emb" not in column_name  # ignore lists of floats that may be embeddings
        and (
            (isinstance(feature, Sequence) and feature.feature == Value("float32"))
            or (isinstance(feature, list) and feature[0] == Value("float32"))
        )
        for column_name, feature in features.items()
    ):
        modalities.add("timeseries")
    # other idea: detect datasets with only numerical columns and one timestamp column
    # (and ideally be able to detect dates/timestamps even from a column with string type)

    return modalities


def detect_modalities_from_features(dataset: str) -> set[DatasetModality]:
    """
    Detect modalities of a dataset using the features (column types).

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.

    Raises:
        [~`libcommon.simple_cache.CachedArtifactError`]:
            If the previous step gave an error.
        [~`libcommon.exceptions.PreviousStepFormatError`]:
            If the content of the previous step has not the expected format

    Returns:
        `set[DatasetModality]`: A set of modalities.
    """
    dataset_info_response = get_previous_step_or_raise(kind="dataset-info", dataset=dataset)
    content = dataset_info_response["content"]
    if "dataset_info" not in content or not isinstance(content["dataset_info"], dict):
        raise PreviousStepFormatError("Previous step did not return the expected content: 'dataset_info'.")

    try:
        modalities: set[DatasetModality] = set()
        for config_info in content["dataset_info"].values():
            modalities.update(detect_features_modalities(features=Features.from_dict(config_info["features"])))
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    return modalities


def detect_modalities_from_url_columns(dataset: str) -> set[DatasetModality]:
    """
    Detect modalities of a dataset using the type of URL columns.
    E.g. if a column contains URLs of images.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.

    Raises:
        [~`libcommon.simple_cache.CachedArtifactError`]:
            If the previous step gave an error.
        [~`libcommon.exceptions.PreviousStepFormatError`]:
            If the content of the previous step has not the expected format

    Returns:
        `set[DatasetModality]`: A set of modalities.
    """
    config_names_response = get_previous_step_or_raise(kind="dataset-split-names", dataset=dataset)
    content = config_names_response["content"]
    if "splits" not in content and not isinstance(content["splits"], list):
        raise PreviousStepFormatError("Previous step did not return the expected content: 'splits'.")

    try:
        for split_item in content["splits"][:10]:  # no need to check all the configs
            config = split_item["config"]
            split = split_item["split"]
            try:
                response = get_response(kind="split-image-url-columns", dataset=dataset, config=config, split=split)
            except CachedArtifactNotFoundError:
                logging.debug("No response found in previous step for this dataset: 'split-image-url-columns'.")
                continue
            if response["http_status"] != HTTPStatus.OK:
                logging.debug(f"Previous step gave an error: {response['http_status']}.")
                continue
            else:
                try:
                    if response["content"]["columns"]:
                        return {"image"}
                except Exception as e:
                    raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    return set()


# from https://developer.mozilla.org/en-US/docs/Web/Media/Formats/Image_types
IMAGE_EXTENSIONS = {
    ".apng",
    ".avif",
    ".gif",
    ".jpg",
    ".jpeg",
    ".jfif",
    ".pjpeg",
    ".pjp",
    ".png",
    ".svg",
    "webp",
    ".bmp",
    ".ico",
    ".cur",
    ".tif",
    ".tiff",
}
# from https://developer.mozilla.org/en-US/docs/Web/Media/Formats/Containers#browser_compatibility + others
AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".mp3",
    ".m4a",
    ".oga",
    ".wav",
    # other audio formats
    ".weba",
    ".opus",
    ".spx",
    ".wma",
    ".aiff",
    ".ape",
    ".mka",
    ".wv",
    ".tak",
}
AUDIO_BUT_COULD_ALSO_BE_VIDEO_EXTENSIONS = {
    ".ogg",
}
VIDEO_EXTENSIONS = {
    ".m4v",
    ".m4p",
    ".ogv",
    ".mov",
    ".mkv",
    # other video formats
    ".avi",
    ".wmv",
    ".flv",
}
VIDEO_BUT_COULD_ALSO_BE_AUDIO_EXTENSIONS = {".3gp", ".mpg", ".mpeg", ".mp4", ".webm"}
GEOSPATIAL_EXTENSIONS = {
    # vectorial
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".cpg",
    ".kml",
    ".kmz",
    ".gpx",
    ".geojson",
    ".topojson",
    ".gml",
    ".geoparquet",
    ".fgb",
    # raster
    ".img",
    ".bil",
    ".bip",
    ".bsq",
    # geotiff uses .tif or .tiff, but better to just show "image" modality
    # than wrongly put "geospatial" if it only contains tif images
    # ".tif",
    # ".tiff",
    # vectorial or raster
    ".gpkg",
    ".mbtiles",
    ".pmtiles",
}
_3D_EXTENSIONS = {
    # from https://docs.unity3d.com/Manual/3D-formats.html
    ".fbx",
    ".dae",
    ".dxf",
    ".obj",
    # other 3D formats
    ".stl",
    ".ply",
    ".gltf",
    ".glb",
    ".usdz",
}
TEXT_EXTENSIONS = {
    ".txt",
}
MULTI_ROWS_EXTENSIONS = {".parquet", ".csv", ".json", ".jsonl", ".arrow"}
ALL_EXTENSIONS = (
    IMAGE_EXTENSIONS
    | AUDIO_EXTENSIONS
    | AUDIO_BUT_COULD_ALSO_BE_VIDEO_EXTENSIONS
    | VIDEO_EXTENSIONS
    | VIDEO_BUT_COULD_ALSO_BE_AUDIO_EXTENSIONS
    | GEOSPATIAL_EXTENSIONS
    | _3D_EXTENSIONS
    | TEXT_EXTENSIONS
    | MULTI_ROWS_EXTENSIONS
)


def detect_modalities_from_filetypes(dataset: str) -> set[DatasetModality]:
    """
    Detect modalities of a dataset using the repository file extensions.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.

    Raises:
        [~`libcommon.simple_cache.CachedArtifactError`]:
            If the previous step gave an error.
        [~`libcommon.exceptions.PreviousStepFormatError`]:
            If the content of the previous step has not the expected format

    Returns:
        `set[DatasetModality]`: A set of modalities.
    """
    dataset_filetypes_response = get_previous_step_or_raise(kind="dataset-filetypes", dataset=dataset)
    content = dataset_filetypes_response["content"]
    if "filetypes" not in content or not isinstance(content["filetypes"], list):
        raise PreviousStepFormatError("Previous step did not return the expected content: 'filetypes'.")

    try:
        modalities: set[DatasetModality] = set()
        total_count = sum(
            filetype["count"] for filetype in content["filetypes"] if filetype["extension"] in ALL_EXTENSIONS
        )
        has_multi_rows_files = any(
            filetype["count"] for filetype in content["filetypes"] if filetype["extension"] in MULTI_ROWS_EXTENSIONS
        )
        min_count = round(0.1 * total_count)  # ignore files that are <10% of the data files to avoid false positives
        min_count_for_image = (
            10 if has_multi_rows_files else 1
        )  # images are often used as figures in README, so we also add this threshold
        for filetype in content["filetypes"]:
            # we condition by a number of files (filetype["count"] > threshold) to avoid false positives
            if filetype["count"] < min_count:
                continue
            elif filetype["extension"] in IMAGE_EXTENSIONS and filetype["count"] < min_count_for_image:
                continue
            if filetype["extension"] in IMAGE_EXTENSIONS:
                modalities.add("image")
            elif filetype["extension"] in AUDIO_EXTENSIONS | AUDIO_BUT_COULD_ALSO_BE_VIDEO_EXTENSIONS:
                modalities.add("audio")
            elif filetype["extension"] in VIDEO_EXTENSIONS | VIDEO_BUT_COULD_ALSO_BE_AUDIO_EXTENSIONS:
                modalities.add("video")
            elif filetype["extension"] in GEOSPATIAL_EXTENSIONS:
                modalities.add("geospatial")
            elif filetype["extension"] in _3D_EXTENSIONS:
                modalities.add("3d")
            elif filetype["extension"] in TEXT_EXTENSIONS:
                modalities.add("text")
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    return modalities


def compute_modalities_response(dataset: str) -> DatasetModalitiesResponse:
    """
    Get the response of 'dataset-modalities' for one specific dataset on huggingface.co.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.

    Raises:
        [~`libcommon.exceptions.PreviousStepFormatError`]:
            If the content of the previous step has not the expected format

    Returns:
        `tuple[DatasetModalitiesResponse, float]`: An object with the modalities_response and the progress.
    """
    logging.info(f"compute 'dataset-modalities' for {dataset=}")

    modalities: set[DatasetModality] = set()
    try:
        modalities.update(detect_modalities_from_features(dataset))
    except PreviousStepFormatError:
        raise
    except Exception:
        logging.info(f"failed to detect modalities from features of {dataset=}")
        pass

    try:
        modalities.update(detect_modalities_from_url_columns(dataset))
    except PreviousStepFormatError:
        raise
    except Exception:
        logging.info(f"failed to detect modalities from file types of {dataset=}")
        pass

    try:
        modalities.update(detect_modalities_from_filetypes(dataset))
    except PreviousStepFormatError:
        raise
    except Exception:
        logging.info(f"failed to detect modalities from file types of {dataset=}")
        pass

    return DatasetModalitiesResponse(
        {
            "modalities": sorted(modalities),
        }
    )


class DatasetModalitiesJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-modalities"

    def compute(self) -> CompleteJobResult:
        response_content = compute_modalities_response(dataset=self.dataset)
        return CompleteJobResult(response_content)
