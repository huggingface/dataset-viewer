# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from pathlib import Path
from typing import Optional

from datasets import Audio, Dataset, Features, Image, Sequence, Value
from libs.libcommon.tests.statistics_dataset import LONG_TEXTS, all_nan_column


def long_text_column() -> list[str]:
    return LONG_TEXTS.split("\n")


def long_text_nan_column() -> list[Optional[str]]:
    texts = long_text_column()
    for i in range(0, len(texts), 7):
        texts[i] = None  # type: ignore
    return texts  # type: ignore


statistics_string_text_dataset = Dataset.from_dict(
    {
        "string_text__column": long_text_column(),
        "string_text__nan_column": long_text_nan_column(),
        "string_text__large_string_column": long_text_column(),
        "string_text__large_string_nan_column": long_text_nan_column(),
    },
    features=Features(
        {
            "string_text__column": Value("string"),
            "string_text__nan_column": Value("string"),
            "string_text__large_string_column": Value("large_string"),
            "string_text__large_string_nan_column": Value("large_string"),
        }
    ),
)


# we don't support dicts of lists
statistics_not_supported_dataset = Dataset.from_dict(
    {
        "list__sequence_dict_column": [
            [{"author": "cat", "content": "mouse", "likes": 5}],
            [{"author": "cat", "content": "mouse", "likes": 5}, {"author": "cat", "content": "mouse", "likes": 5}],
            [{"author": "cat", "content": "mouse", "likes": 5}, {"author": "cat", "content": "mouse", "likes": 5}],
            [{"author": "cat", "content": "mouse", "likes": 5}, {"author": "cat", "content": "mouse", "likes": 5}],
            [{"author": "cat", "content": "mouse", "likes": 5}, {"author": "cat", "content": "mouse", "likes": 5}],
            [{"author": "cat", "content": "mouse", "likes": 5}, {"author": "cat", "content": "mouse", "likes": 5}, {}],
            [{"author": "cat", "content": "mouse", "likes": 5}, {"author": "cat", "content": "mouse", "likes": 5}, {}],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [],
            [],
        ],
        "list__sequence_dict_nan_column": [
            None,
            None,
            None,
            None,
            [{"author": "cat", "content": "mouse", "likes": 5}, {"author": "cat", "content": "mouse", "likes": 5}],
            [{"author": "cat", "content": "mouse", "likes": 5}, {"author": "cat", "content": "mouse", "likes": 5}, {}],
            [{"author": "cat", "content": "mouse", "likes": 5}, {"author": "cat", "content": "mouse", "likes": 5}, {}],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
                {"author": "cat", "content": "mouse", "likes": 5},
            ],
            [],
            [],
        ],
        "list__sequence_dict_all_nan_column": all_nan_column(20),
    },
    features=Features(
        {
            "list__sequence_dict_column": Sequence(
                {"author": Value("string"), "content": Value("string"), "likes": Value("int32")}
            ),
            "list__sequence_dict_nan_column": Sequence(
                {"author": Value("string"), "content": Value("string"), "likes": Value("int32")}
            ),
            "list__sequence_dict_all_nan_column": Sequence(
                {"author": Value("string"), "content": Value("string"), "likes": Value("int32")}
            ),
        }
    ),
)


audio_dataset = Dataset.from_dict(
    {
        "audio": [
            str(Path(__file__).resolve().parent / "data" / "audio" / "audio_1.wav"),
            str(Path(__file__).resolve().parent / "data" / "audio" / "audio_2.wav"),
            str(Path(__file__).resolve().parent / "data" / "audio" / "audio_3.wav"),
            str(Path(__file__).resolve().parent / "data" / "audio" / "audio_4.wav"),
        ],
        "audio_nan": [
            str(Path(__file__).resolve().parent / "data" / "audio" / "audio_1.wav"),
            None,
            str(Path(__file__).resolve().parent / "data" / "audio" / "audio_3.wav"),
            None,
        ],
        "audio_all_nan": [None, None, None, None],
    },
    features=Features(
        {
            "audio": Audio(sampling_rate=1600, decode=False),
            "audio_nan": Audio(sampling_rate=1600, decode=False),
            "audio_all_nan": Audio(sampling_rate=1600, decode=False),
        }
    ),
)


image_dataset = Dataset.from_dict(
    {
        "image": [
            str(Path(__file__).resolve().parent / "data" / "image" / "image_1.jpg"),
            str(Path(__file__).resolve().parent / "data" / "image" / "image_2.png"),
            str(Path(__file__).resolve().parent / "data" / "image" / "image_3.jpg"),
            str(Path(__file__).resolve().parent / "data" / "image" / "image_4.jpg"),
        ],
        "image_nan": [
            str(Path(__file__).resolve().parent / "data" / "image" / "image_1.jpg"),
            None,
            str(Path(__file__).resolve().parent / "data" / "image" / "image_3.jpg"),
            None,
        ],
        "image_all_nan": [None, None, None, None],
    },
    features=Features(
        {
            "image": Image(decode=False),
            "image_nan": Image(decode=False),
            "image_all_nan": Image(decode=False),
        }
    ),
)
