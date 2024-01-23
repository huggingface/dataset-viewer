# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Union

from datasets import Audio, Features, Image
from datasets.features.features import FeatureType, Sequence

from libcommon.dtos import FeatureItem


class InvalidFirstRowsError(ValueError):
    pass


VisitPath = list[Union[str, Literal[0]]]


@dataclass
class AssetUrlPath:
    feature_type: Literal["Audio", "Image"]
    path: VisitPath


# invert to_features_list
def to_features_dict(features: list[FeatureItem]) -> Features:
    return Features({feature_item["name"]: feature_item["type"] for feature_item in features})


def _visit(
    feature: FeatureType, func: Callable[[FeatureType, VisitPath], Optional[FeatureType]], visit_path: VisitPath = []
) -> FeatureType:
    """Visit a (possibly nested) feature.

    Args:
        feature (FeatureType): the feature type to be checked
    Returns:
        visited feature (FeatureType)
    """
    if isinstance(feature, dict):
        out = func({k: _visit(f, func, visit_path + [k]) for k, f in feature.items()}, visit_path)
    elif isinstance(feature, (list, tuple)):
        out = func([_visit(feature[0], func, visit_path + [0])], visit_path)
    elif isinstance(feature, Sequence):
        out = func(Sequence(_visit(feature.feature, func, visit_path + [0]), length=feature.length), visit_path)
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
                asset_url_paths.append(AssetUrlPath(feature_type="Audio", path=visit_path + [0]))

        _visit(feature, classify, [column])
    return asset_url_paths


class URLSigner(ABC):
    @abstractmethod
    def sign_url(self, url: str) -> str:
        pass

    def _sign_asset_url_path(self, cell: Any, asset_url_path: AssetUrlPath) -> Any:
        if len(asset_url_path.path) == 0:
            if not isinstance(cell, dict):
                raise InvalidFirstRowsError("Expected the cell to be a dict")
            src = cell.get("src")
            if not isinstance(src, str):
                raise InvalidFirstRowsError('Expected cell["src"] to be a string')
            cell["src"] = self.sign_url(url=src)
            # ^ sign the url in place
        else:
            key = asset_url_path.path.pop(0)
            if key == 0:
                # it's a list, we have to sign each element
                if not isinstance(cell, list):
                    raise InvalidFirstRowsError("Expected the cell to be a list")
                for cell_item in cell:
                    self._sign_asset_url_path(cell=cell_item, asset_url_path=asset_url_path)
            else:
                # it's a dict, we have to sign the value of the key
                if not isinstance(cell, dict):
                    raise InvalidFirstRowsError("Expected the cell to be a dict")
                cell[key] = self._sign_asset_url_path(cell=cell[key], asset_url_path=asset_url_path)

    def sign_urls_in_first_rows_in_place(self, first_rows: Any) -> None:
        if not isinstance(first_rows, dict):
            raise InvalidFirstRowsError("Expected response to be a dict")
        # parse the features to find the paths to assets URLs
        features_list = first_rows.get("features")
        if not isinstance(features_list, list):
            raise InvalidFirstRowsError('Expected response["features"] a list')
        features_dict = to_features_dict(features_list)
        features = Features.from_dict(features_dict)
        asset_url_paths = get_asset_url_paths(features)
        if not asset_url_paths:
            return
        # sign the URLs
        row_items = first_rows.get("rows")
        if not isinstance(row_items, list):
            raise InvalidFirstRowsError('Expected response["rows"] to be a list')
        for row_item in row_items:
            if not isinstance(row_item, dict):
                raise InvalidFirstRowsError('Expected response["rows"][i] to be a dict')
            row = row_item.get("row")
            if not isinstance(row, dict):
                raise InvalidFirstRowsError('Expected response["rows"][i]["row"] to be a dict')
            for asset_url_path in asset_url_paths:
                self._sign_asset_url_path(cell=row, asset_url_path=asset_url_path)
