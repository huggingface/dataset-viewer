from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from datasets import (
    Array2D,
    Array3D,
    Array4D,
    Array5D,
    Audio,
    ClassLabel,
    Dataset,
    Features,
    Image,
    Sequence,
    Translation,
    TranslationVariableLanguages,
    Value,
)
from datasets.features.features import FeatureType


def create_dataset(content: Any, feature_type: FeatureType = None) -> Dataset:
    return (
        Dataset.from_dict({"col": [content]})
        if feature_type is None
        else Dataset.from_dict({"col": [content]}, features=Features({"col": feature_type}))
    )


@pytest.fixture(scope="session")
def datasets() -> Dict[str, Dataset]:
    sampling_rate = 16_000
    data = {
        "class_label": ("positive", ClassLabel(names=["negative", "positive"])),
        "dict": ({"a": 0}, None),
        "list": ([{"a": 0}], None),
        "sequence_simple": ([0], None),
        "sequence": ([{"a": 0}], Sequence(feature={"a": Value(dtype="int64")})),
        "sequence_audio": (
            [
                {"array": [0.1, 0.2, 0.3], "sampling_rate": 16_000},
            ],
            Sequence(feature=Audio()),
        ),
        "array2d": (np.zeros((2, 2), dtype="float32"), Array2D(shape=(2, 2), dtype="float32")),
        "array3d": (np.zeros((2, 2, 2), dtype="float32"), Array3D(shape=(2, 2, 2), dtype="float32")),
        "array4d": (np.zeros((2, 2, 2, 2), dtype="float32"), Array4D(shape=(2, 2, 2, 2), dtype="float32")),
        "array5d": (np.zeros((2, 2, 2, 2, 2), dtype="float32"), Array5D(shape=(2, 2, 2, 2, 2), dtype="float32")),
        "audio": ({"array": [0.1, 0.2, 0.3], "sampling_rate": sampling_rate}, Audio(sampling_rate=sampling_rate)),
        "image": (str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"), Image()),
        "translation": ({"en": "the cat", "fr": "le chat"}, Translation(languages=["en", "fr"])),
        "translation_variable_languages": (
            {"en": "the cat", "fr": ["le chat", "la chatte"]},
            TranslationVariableLanguages(languages=["en", "fr"]),
        ),
    }
    return {k: create_dataset(*args) for k, args in data.items()}
