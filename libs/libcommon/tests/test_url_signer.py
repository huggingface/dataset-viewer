# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from collections.abc import Mapping

import pytest

from libcommon.url_signer import AssetUrlPath, get_asset_url_paths, to_features_dict
from libcommon.viewer_utils.features import to_features_list

from .constants import DATASETS_NAMES
from .types import DatasetFixture


@pytest.mark.parametrize("dataset_name", DATASETS_NAMES)
def test_to_features_dict(datasets_fixtures: Mapping[str, DatasetFixture], dataset_name: str) -> None:
    datasets_fixture = datasets_fixtures[dataset_name]
    features = datasets_fixture.dataset.features
    features_list = to_features_list(features)
    features_dict = to_features_dict(features_list)
    assert isinstance(features_dict, dict)
    assert len(features_dict) > 0
    assert features.to_dict() == features_dict


@pytest.mark.parametrize("dataset_name", DATASETS_NAMES)
def test_get_asset_url_paths(datasets_fixtures: Mapping[str, DatasetFixture], dataset_name: str) -> None:
    dataset_fixture = datasets_fixtures[dataset_name]
    asset_url_paths = get_asset_url_paths(dataset_fixture.dataset.features)
    assert isinstance(asset_url_paths, list)
    if dataset_name in {"audio", "audio_ogg"}:
        assert len(asset_url_paths) == 1
        assert asset_url_paths[0] == AssetUrlPath(feature_type="Audio", path=["col", 0])
    elif dataset_name in {"images_list", "images_sequence"}:
        assert len(asset_url_paths) == 1
        assert asset_url_paths[0] == AssetUrlPath(feature_type="Image", path=["col", 0])
    elif dataset_name in {"image"}:
        assert len(asset_url_paths) == 1
        assert asset_url_paths[0] == AssetUrlPath(feature_type="Image", path=["col"])
    elif dataset_name in {"audios_list", "audios_sequence"}:
        assert len(asset_url_paths) == 1
        assert asset_url_paths[0] == AssetUrlPath(feature_type="Audio", path=["col", 0, 0])
    elif dataset_name in {"dict_of_audios_and_images"}:
        assert len(asset_url_paths) == 2
        assert asset_url_paths == [
            AssetUrlPath(feature_type="Image", path=["col", "b", 0]),
            AssetUrlPath(feature_type="Audio", path=["col", "c", "ca", 0, 0]),
        ]
    else:
        assert len(asset_url_paths) == 0
