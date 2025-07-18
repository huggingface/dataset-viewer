# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Union

from datasets import Audio, Features, Image, Pdf, Video
from datasets.features.features import FeatureType, LargeList, List

from libcommon.cloudfront import CloudFrontSigner
from libcommon.dtos import FeatureItem
from libcommon.viewer_utils.asset import replace_dataset_git_revision_placeholder


class InvalidFirstRowsError(ValueError):
    pass


VisitPath = list[Union[str, Literal[0]]]


@dataclass
class AssetUrlPath:
    feature_type: Literal["Audio", "Image", "Video", "Pdf"]
    path: VisitPath

    def enter(self) -> "AssetUrlPath":
        if len(self.path) == 0:
            raise ValueError("Cannot enter an empty path")
        return AssetUrlPath(feature_type=self.feature_type, path=self.path[1:])


# invert to_features_list
def to_features_dict(features: list[FeatureItem]) -> Features:
    return {feature_item["name"]: feature_item["type"] for feature_item in features}


def _visit(
    feature: FeatureType, func: Callable[[FeatureType, VisitPath], Optional[FeatureType]], visit_path: VisitPath = []
) -> FeatureType:
    """Visit a (possibly nested) feature.

    Args:
        feature (`FeatureType`): the feature type to be checked.

    Returns:
        `FeatureType`: the visited feature.
    """
    if isinstance(feature, dict):
        out = func({k: _visit(f, func, visit_path + [k]) for k, f in feature.items()}, visit_path)
    elif isinstance(feature, (list, tuple)):
        out = func([_visit(feature[0], func, visit_path + [0])], visit_path)
    elif isinstance(feature, List):
        out = func(List(_visit(feature.feature, func, visit_path + [0]), length=feature.length), visit_path)
    elif isinstance(feature, LargeList):
        out = func(LargeList(_visit(feature.feature, func, visit_path + [0])), visit_path)
    else:
        out = func(feature, visit_path)
    return feature if out is None else out


def get_asset_url_paths(features: Features) -> list[AssetUrlPath]:
    asset_url_paths: list[AssetUrlPath] = []
    for column, feature in features.items():

        def classify(feature: FeatureType, visit_path: VisitPath) -> None:
            if isinstance(feature, Image):
                asset_url_paths.append(AssetUrlPath(feature_type="Image", path=visit_path))
            elif isinstance(feature, Audio):
                # for audio we give a list in case there are multiple formats available
                asset_url_paths.append(AssetUrlPath(feature_type="Audio", path=visit_path + [0]))
            elif isinstance(feature, Video):
                asset_url_paths.append(AssetUrlPath(feature_type="Video", path=visit_path))
            elif isinstance(feature, Pdf):
                asset_url_paths.append(AssetUrlPath(feature_type="Pdf", path=visit_path))

        _visit(feature, classify, [column])
    return asset_url_paths


class URLPreparator(ABC):
    def __init__(self, url_signer: Optional[CloudFrontSigner], hf_endpoint: str, assets_base_url: str) -> None:
        self.url_signer = url_signer
        self.hf_endpoint = hf_endpoint
        self.assets_base_url = assets_base_url

    def prepare_url(self, url: str, revision: str) -> str:
        # Set the right revision in the URL e.g.
        # Before: https://datasets-server.huggingface.co/assets/vidore/syntheticDocQA_artificial_intelligence_test/--/{dataset_git_revision}/--/default/test/0/image/image.jpg
        # After:  https://datasets-server.huggingface.co/assets/vidore/syntheticDocQA_artificial_intelligence_test/--/5fe59d7e52732b86d11ee0e9c4a8cdb0e8ba7a6e/--/default/test/0/image/image.jpg
        url = replace_dataset_git_revision_placeholder(url, revision)
        # Sign the URL since the assets require authentication to be accessed
        # Before: https://datasets-server.huggingface.co/assets/vidore/syntheticDocQA_artificial_intelligence_test/--/5fe59d7e52732b86d11ee0e9c4a8cdb0e8ba7a6e/--/default/test/0/image/image.jpg
        # After:  https://datasets-server.huggingface.co/assets/vidore/syntheticDocQA_artificial_intelligence_test/--/5fe59d7e52732b86d11ee0e9c4a8cdb0e8ba7a6e/--/default/test/0/image/image.jpg?Expires=1...4&Signature=E...A__&Key-Pair-Id=K...3
        if self.url_signer and url.startswith(self.assets_base_url):
            url = self.url_signer.sign_url(url)
        # Convert HF URL to HF HTTP URL e.g.
        # Before: hf://datasets/username/dataset_name@5fe59d7e52732b86d11ee0e9c4a8cdb0e8ba7a6e/video.mp4
        # After:  https://huggingface.co/datasets/username/dataset_name/resolve/5fe59d7e52732b86d11ee0e9c4a8cdb0e8ba7a6e/video.mp4
        if url.startswith("hf://"):
            url = url.replace("hf://", self.hf_endpoint + "/").replace("@", "/resolve/")
        return url

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(url_signer={self.url_signer})"

    def _prepare_asset_url_path_in_place(self, cell: Any, asset_url_path: AssetUrlPath, revision: str) -> Any:
        if not cell:
            return cell
        elif len(asset_url_path.path) == 0:
            if not isinstance(cell, dict):
                raise InvalidFirstRowsError("Expected the cell to be a dict")
            for key, value in cell.items():
                if isinstance(value, dict):
                    # if the value is a dict, we have to prepare the URL in it for nested assets
                    self._prepare_asset_url_path_in_place(cell=value, asset_url_path=asset_url_path, revision=revision)
                elif key == "src":
                    src = cell.get(key)
                    if not isinstance(src, str):
                        raise InvalidFirstRowsError(f'Expected cell["{key}"] to be a string')
                    cell[key] = self.prepare_url(src, revision=revision)
            # ^ prepare the url in place
        else:
            key = asset_url_path.path[0]
            if key == 0:
                # it's a list, we have to prepare each element
                if not isinstance(cell, list):
                    raise InvalidFirstRowsError("Expected the cell to be a list")
                for cell_item in cell:
                    self._prepare_asset_url_path_in_place(
                        cell=cell_item, asset_url_path=asset_url_path.enter(), revision=revision
                    )
            else:
                # it's a dict, we have to prepare the value of the key
                if not isinstance(cell, dict):
                    raise InvalidFirstRowsError("Expected the cell to be a dict")
                self._prepare_asset_url_path_in_place(
                    cell=cell[key], asset_url_path=asset_url_path.enter(), revision=revision
                )

    def _get_asset_url_paths_from_first_rows(self, first_rows: Mapping[str, Any]) -> list[AssetUrlPath]:
        # parse the features to find the paths to assets URLs
        features_list = first_rows.get("features")
        if not isinstance(features_list, list):
            raise InvalidFirstRowsError('Expected response["features"] a list')
        features_dict = to_features_dict(features_list)
        features = Features.from_dict(features_dict)
        return get_asset_url_paths(features)

    def prepare_urls_in_first_rows_in_place(self, first_rows: Mapping[str, Any], revision: str) -> None:
        asset_url_paths = self._get_asset_url_paths_from_first_rows(first_rows=first_rows)
        if not asset_url_paths:
            return
        # prepare the URLs (set revision + sign)
        row_items = first_rows.get("rows")
        if not isinstance(row_items, list):
            raise InvalidFirstRowsError('Expected response["rows"] to be a list')
        for row_item in row_items:
            if not isinstance(row_item, dict):
                raise InvalidFirstRowsError('Expected response["rows"][i] to be a dict')
            truncated_cells = row_item.get("truncated_cells")
            if not isinstance(truncated_cells, list) or not all(isinstance(cell, str) for cell in truncated_cells):
                raise InvalidFirstRowsError('Expected response["rows"][i]["truncated_cells"] to be a list of strings')
            row = row_item.get("row")
            if not isinstance(row, dict):
                raise InvalidFirstRowsError('Expected response["rows"][i]["row"] to be a dict')
            for asset_url_path in asset_url_paths:
                if isinstance(asset_url_path.path[0], str) and asset_url_path.path[0] in truncated_cells:
                    # the cell has been truncated, nothing to prepare in it
                    continue
                self._prepare_asset_url_path_in_place(cell=row, asset_url_path=asset_url_path, revision=revision)
