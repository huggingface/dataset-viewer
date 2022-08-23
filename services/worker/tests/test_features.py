import datetime
from typing import Any, Dict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd  # type: ignore
import pyarrow as pa  # type: ignore
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

from worker.features import get_cell_value

from .utils import ASSETS_BASE_URL

# we need to know the correspondence between the feature type and the cell value, in order to:
# - document the API
# - implement the client on the Hub (dataset viewer)


# see https://github.com/huggingface/datasets/blob/a5192964dc4b76ee5c03593c11ee56f29bbd688d/...
#     src/datasets/features/features.py#L1469
# ``FieldType`` can be one of the following:
# - a :class:`datasets.Value` feature specifies a single typed value, e.g. ``int64`` or ``string``
@pytest.mark.parametrize(
    "input_value,input_dtype,output_value,output_dtype",
    [
        # null
        (None, None, None, "null"),
        # bool
        (False, pd.BooleanDtype(), False, "bool"),
        # int8
        (-7, pd.Int8Dtype(), -7, "int8"),
        # int16
        (-7, pd.Int16Dtype(), -7, "int16"),
        # int32
        (-7, pd.Int32Dtype(), -7, "int32"),
        # int64
        (-7, pd.Int64Dtype(), -7, "int64"),
        # uint8
        (7, pd.UInt8Dtype(), 7, "uint8"),
        # uint16
        (7, pd.UInt16Dtype(), 7, "uint16"),
        # uint32
        (7, pd.UInt32Dtype(), 7, "uint32"),
        # uint64
        (7, pd.UInt64Dtype(), 7, "uint64"),
        # float16
        (-3.14, np.float16, np.float16(-3.14), "float16"),
        # ^ TODO: is it a datasets bug?
        # float32 (alias float)
        (-3.14, np.float32, np.float32(-3.14), "float32"),
        # ^ TODO: is it a datasets bug?
        # float64 (alias double)
        (-3.14, np.float64, -3.14, "float64"),
        # time32[(s|ms)]
        # TODO
        # time64[(us|ns)]
        # (time(1, 1, 1), None, datetime.datetime(1, 1, 1), "time64[us]"),
        # ^ TODO: add after https://github.com/huggingface/datasets/issues/4620 is fixed
        # timestamp[(s|ms|us|ns)]
        (pd.Timestamp(2020, 1, 1), None, datetime.datetime(2020, 1, 1, 0, 0), "timestamp[ns]"),
        (
            pd.Timestamp(1513393355.5, unit="s"),
            None,
            datetime.datetime(2017, 12, 16, 3, 2, 35, 500000),
            "timestamp[ns]",
        ),
        (
            pd.Timestamp(1513393355500, unit="ms"),
            None,
            datetime.datetime(2017, 12, 16, 3, 2, 35, 500000),
            "timestamp[ns]",
        ),
        # timestamp[(s|ms|us|ns), tz=(tzstring)]
        (
            pd.Timestamp(year=2020, month=1, day=1, tz="US/Pacific"),
            None,
            datetime.datetime(2020, 1, 1, 0, 0, tzinfo=ZoneInfo("US/Pacific")),
            "timestamp[ns, tz=US/Pacific]",
        ),
        # date32
        # TODO
        # date64
        # TODO
        # duration[(s|ms|us|ns)]
        # TODO
        # decimal128(precision, scale)
        # TODO
        # decimal256(precision, scale)
        # TODO
        # binary
        # TODO
        # large_binary
        # TODO
        # string
        ("a string", pd.StringDtype(), "a string", "string"),
        # large_string
        # TODO
    ],
)
def test_value(input_value, input_dtype, output_value, output_dtype) -> None:
    if input_dtype == "datetime64[ns]":
        a = pa.array(
            [
                datetime.datetime(2022, 7, 4, 3, 2, 1),
            ],
            type=pa.date64(),
        )
        dataset = Dataset.from_buffer(a.to_buffer())
    else:
        df = pd.DataFrame({"feature_name": [input_value]}, dtype=input_dtype)
        dataset = Dataset.from_pandas(df)
    feature = dataset.features["feature_name"]
    assert feature._type == "Value"
    assert feature.dtype == output_dtype
    value = get_cell_value(
        "dataset", "config", "split", 7, dataset[0]["feature_name"], "feature_name", feature, ASSETS_BASE_URL
    )
    assert value == output_value


@pytest.mark.parametrize(
    "dataset_type,output_value,output_type",
    [
        # - a :class:`datasets.ClassLabel` feature specifies a field with a predefined set of classes
        #   which can have labels associated to them and will be stored as integers in the dataset
        ("class_label", 1, "ClassLabel"),
        # - a python :obj:`dict` which specifies that the field is a nested field containing a mapping of sub-fields
        #   to sub-fields features. It's possible to have nested fields of nested fields in an arbitrary manner
        ("dict", {"a": 0}, {"a": Value(dtype="int64", id=None)}),
        # - a python :obj:`list` or a :class:`datasets.Sequence` specifies that the field contains a list of objects.
        #    The python :obj:`list` or :class:`datasets.Sequence` should be provided with a single sub-feature as an
        #    example of the feature type hosted in this list
        #   <Tip>
        #   A :class:`datasets.Sequence` with a internal dictionary feature will be automatically converted into a
        #   dictionary of lists. This behavior is implemented to have a compatilbity layer with the TensorFlow Datasets
        #   library but may be un-wanted in some cases. If you don't want this behavior, you can use a python
        #   :obj:`list` instead of the :class:`datasets.Sequence`.
        #   </Tip>
        ("list", [{"a": 0}], [{"a": Value(dtype="int64", id=None)}]),
        ("sequence_simple", [0], "Sequence"),
        ("sequence", {"a": [0]}, "Sequence"),
        # (
        #     "sequence_audio"
        #     # ^ corner case: an Audio in a Sequence
        #     [{"path": None, "array": np.array([0.09997559, 0.19998169, 0.29998779]), "sampling_rate": 16_000}],
        #     "Sequence"
        # ),
        # - a :class:`Array2D`, :class:`Array3D`, :class:`Array4D` or :class:`Array5D` feature for multidimensional
        #   arrays
        ("array2d", [[0.0, 0.0], [0.0, 0.0]], "Array2D"),
        ("array3d", [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]], "Array3D"),
        (
            "array4d",
            [
                [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            ],
            "Array4D",
        ),
        (
            "array5d",
            [
                [
                    [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                ],
                [
                    [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                ],
            ],
            "Array5D",
        ),
        # - an :class:`Audio` feature to store the absolute path to an audio file or a dictionary with the relative
        #   path to an audio file ("path" key) and its bytes content ("bytes" key). This feature extracts the audio
        #   data.
        (
            "audio",
            [
                {
                    "src": "http://localhost/assets/dataset/--/config/split/7/col/audio.mp3",
                    "type": "audio/mpeg",
                },
                {
                    "src": "http://localhost/assets/dataset/--/config/split/7/col/audio.wav",
                    "type": "audio/wav",
                },
            ],
            "Audio",
        ),
        # - an :class:`Image` feature to store the absolute path to an image file, an :obj:`np.ndarray` object, a
        #   :obj:`PIL.Image.Image` object or a dictionary with the relative path to an image file ("path" key) and
        #   its bytes content ("bytes" key). This feature extracts the image data.
        ("image", "http://localhost/assets/dataset/--/config/split/7/col/image.jpg", "Image"),
        # - :class:`datasets.Translation` and :class:`datasets.TranslationVariableLanguages`, the two features
        #   specific to Machine Translation
        ("translation", {"en": "the cat", "fr": "le chat"}, "Translation"),
        (
            "translation_variable_languages",
            {"language": ["en", "fr", "fr"], "translation": ["the cat", "la chatte", "le chat"]},
            "TranslationVariableLanguages",
        ),
    ],
)
def test_others(dataset_type: str, output_value: Any, output_type: Any, datasets: Dict[str, Dataset]) -> None:
    dataset = datasets[dataset_type]
    feature = dataset.features["col"]
    if type(output_type) in [list, dict]:
        assert feature == output_type
    else:
        assert feature._type == output_type
    value = get_cell_value("dataset", "config", "split", 7, dataset[0]["col"], "col", feature, ASSETS_BASE_URL)
    assert value == output_value
