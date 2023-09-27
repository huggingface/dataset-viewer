# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import datetime
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
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

# from GLUE dataset, "ax" subset
LONG_TEXTS = """The cat sat on the mat.
The cat did not sit on the mat.
When you've got no snow, it's really hard to learn a snow sport so we looked at all the different ways I could mimic being on snow without actually being on snow.
When you've got snow, it's really hard to learn a snow sport so we looked at all the different ways I could mimic being on snow without actually being on snow.
Out of the box, Ouya supports media apps such as Twitch.tv and XBMC media player.
Out of the box, Ouya doesn't support media apps such as Twitch.tv and XBMC media player.
Out of the box, Ouya supports media apps such as Twitch.tv and XBMC media player.
Out of the box, Ouya supports Twitch.tv and XBMC media player.
Considering this definition, it is surprising to find frequent use of sarcastic language in opinionated user generated content.
Considering this definition, it is not surprising to find frequent use of sarcastic language in opinionated user generated content.
The new gaming console is affordable.
The new gaming console is unaffordable.
Brexit is an irreversible decision, Sir Mike Rake, the chairman of WorldPay and ex-chairman of BT group, said as calls for a second EU referendum were sparked last week.
Brexit is a reversible decision, Sir Mike Rake, the chairman of WorldPay and ex-chairman of BT group, said as calls for a second EU referendum were sparked last week.
We built our society on unclean energy.
We built our society on clean energy.
Pursuing a strategy of nonviolent protest, Gandhi took the administration by surprise and won concessions from the authorities.
Pursuing a strategy of violent protest, Gandhi took the administration by surprise and won concessions from the authorities.
Pursuing a strategy of nonviolent protest, Gandhi took the administration by surprise and won concessions from the authorities.
Pursuing a strategy of protest, Gandhi took the administration by surprise and won concessions from the authorities.
And if both apply, they are essentially impossible.
And if both apply, they are essentially possible.
Writing Java is not too different from programming with handcuffs.
Writing Java is similar to programming with handcuffs.
The market is about to get harder, but not impossible to navigate.
The market is about to get harder, but possible to navigate.
Even after now finding out that it's animal feed, I won't ever stop being addicted to Flamin' Hot Cheetos.
Even after now finding out that it's animal feed, I will never stop being addicted to Flamin' Hot Cheetos.
He did not disagree with the party's position, but felt that if he resigned, his popularity with Indians would cease to stifle the party's membership.
He agreed with the party's position, but felt that if he resigned, his popularity with Indians would cease to stifle the party's membership.
If the pipeline tokenization scheme does not correspond to the one that was used when a model was created, a negative impact on the pipeline results would be expected.
If the pipeline tokenization scheme does not correspond to the one that was used when a model was created, a negative impact on the pipeline results would not be unexpected.
If the pipeline tokenization scheme does not correspond to the one that was used when a model was created, a negative impact on the pipeline results would be expected.
If the pipeline tokenization scheme does not correspond to the one that was used when a model was created, it would be expected to negatively impact the pipeline results.
If the pipeline tokenization scheme does not correspond to the one that was used when a model was created, a negative impact on the pipeline results would be expected.
If the pipeline tokenization scheme does not correspond to the one that was used when a model was created, it would not be unexpected for it to negatively impact the pipeline results.
The water is too hot.
The water is too cold.
Falcon Heavy is the largest rocket since NASA's Saturn V booster, which was used for the Moon missions in the 1970s.
Falcon Heavy is the smallest rocket since NASA's Saturn V booster, which was used for the Moon missions in the 1970s.
Adenoiditis symptoms often persist for ten or more days, and often include pus-like discharge from nose.
Adenoiditis symptoms often pass within ten days or less, and often include pus-like discharge from nose.
In example (1) it is quite straightforward to see the exaggerated positive sentiment used in order to convey strong negative feelings.
In example (1) it is quite difficult to see the exaggerated positive sentiment used in order to convey strong negative feelings.
In example (1) it is quite straightforward to see the exaggerated positive sentiment used in order to convey strong negative feelings.
In example (1) it is quite easy to see the exaggerated positive sentiment used in order to convey strong negative feelings.
In example (1) it is quite straightforward to see the exaggerated positive sentiment used in order to convey strong negative feelings.
In example (1) it is quite important to see the exaggerated positive sentiment used in order to convey strong negative feelings.
Some dogs like to scratch their ears.
Some animals like to scratch their ears.
Cruz has frequently derided as "amnesty" various plans that confer legal status or citizenship on people living in the country illegally.
Cruz has frequently derided as "amnesty" various bills that confer legal status or citizenship on people living in the country illegally.
Most of the graduates of my program have moved on to other things because the jobs suck.
Some of the graduates of my program have moved on to other things because the jobs suck.
In many developed areas, human activity has changed the form of river channels, altering magnitudes and frequencies of flooding.
In many areas, human activity has changed the form of river channels, altering magnitudes and frequencies of flooding.
We consider some context words as positive examples and sample negatives at random from the dictionary.
We consider some words as positive examples and sample negatives at random from the dictionary.
We consider some context words as positive examples and sample negatives at random from the dictionary.
We consider all context words as positive examples and sample many negatives at random from the dictionary.
We consider some context words as positive examples and sample negatives at random from the dictionary.
We consider many context words as positive examples and sample negatives at random from the dictionary.
We consider all context words as positive examples and sample negatives at random from the dictionary.
We consider all words as positive examples and sample negatives at random from the dictionary.
All dogs like to scratch their ears.
All animals like to scratch their ears.
Cruz has frequently derided as "amnesty" any plan that confers legal status or citizenship on people living in the country illegally.
Cruz has frequently derided as "amnesty" any bill that confers legal status or citizenship on people living in the country illegally.
Most of the graduates of my program have moved on to other things because the jobs suck.
None of the graduates of my program have moved on to other things because the jobs suck.
Most of the graduates of my program have moved on to other things because the jobs suck.
All of the graduates of my program have moved on to other things because the jobs suck.
In all areas, human activity has changed the form of river channels, altering magnitudes and frequencies of flooding.
In all developed areas, human activity has changed the form of river channels, altering magnitudes and frequencies of flooding.
Tom and Adam were whispering in the theater.
Tom and Adam were whispering quietly in the theater.
Tom and Adam were whispering in the theater.
Tom and Adam were whispering loudly in the theater.
Prior to the dance, which is voluntary, students are told to fill out a card by selecting five people they want to dance with.
Prior to the dance, which is voluntary, students are told to fill out a card by selecting five different people they want to dance with.
Notifications about Farmville and other crap had become unbearable, then the shift to the non-chronological timeline happened and the content from your friends started to be replaced by ads and other cringy wannabe-viral campaigns.
Notifications about Farmville and other crappy apps had become unbearable, then the shift to the non-chronological timeline happened and the content from your friends started to be replaced by ads and other cringy wannabe-viral campaigns.
Chicago City Hall is the official seat of government of the City of Chicago.
Chicago City Hall is the official seat of government of Chicago.
The question generation aspect is unique to our formulation, and corresponds roughly to identifying what semantic role labels are present in previous formulations of the task.
The question generation aspect is unique to our formulation, and corresponds roughly to identifying what semantic role labels are present in previous other formulations of the task.
John ate pasta for dinner.
John ate pasta for supper.
John ate pasta for dinner.
John ate pasta for breakfast.
House Speaker Paul Ryan was facing problems from fellow Republicans dissatisfied with his leadership.
House Speaker Paul Ryan was facing problems from fellow Republicans unhappy with his leadership.
House Speaker Paul Ryan was facing problems uniquely from fellow Republicans dissatisfied with his leadership.
House Speaker Paul Ryan was facing problems uniquely from fellow Republicans supportive of his leadership.
I can actually see him climbing into a Lincoln saying this.
I can actually see him getting into a Lincoln saying this.
I can actually see him climbing into a Lincoln saying this.
I can actually see him climbing into a Mazda saying this.
The villain is the character who tends to have a negative effect on other characters.
The villain is the character who tends to have a negative impact on other characters.
"""  # noqa: E501


def long_text_column() -> list[str]:
    return LONG_TEXTS.split("\n")


def long_text_nan_column() -> list[Optional[str]]:
    texts = long_text_column()
    for i in range(0, len(texts), 7):
        texts[i] = None  # type: ignore
    return texts  # type: ignore


def value(content: Any, dtype: Any) -> Dataset:
    return Dataset.from_pandas(pd.DataFrame({"col": [content]}, dtype=dtype))


def other(content: Any, feature_type: Optional[FeatureType] = None) -> Dataset:
    if feature_type:
        features = Features({"col": feature_type})
        return Dataset.from_dict({"col": [content]}, features=features)
    else:
        return Dataset.from_dict({"col": [content]})


@pytest.fixture(scope="session")
def datasets() -> Mapping[str, Dataset]:
    sampling_rate = 16_000
    return {
        # Value feature
        "null": value(None, None),
        "bool": value(False, pd.BooleanDtype()),
        "int8": value(-7, pd.Int8Dtype()),
        "int16": value(-7, pd.Int16Dtype()),
        "int32": value(-7, pd.Int32Dtype()),
        "int64": value(-7, pd.Int64Dtype()),
        "uint8": value(7, pd.UInt8Dtype()),
        "uint16": value(7, pd.UInt16Dtype()),
        "uint32": value(7, pd.UInt32Dtype()),
        "uint64": value(7, pd.UInt64Dtype()),
        "float16": value(-3.14, np.float16),
        "float32": value(-3.14, np.float32),
        "float64": value(-3.14, np.float64),
        "time": value(datetime.time(1, 1, 1), None),
        "timestamp_1": value(pd.Timestamp(2020, 1, 1), None),
        "timestamp_2": value(pd.Timestamp(1513393355.5, unit="s"), None),
        "timestamp_3": value(pd.Timestamp(1513393355500, unit="ms"), None),
        "timestamp_tz": value(pd.Timestamp(year=2020, month=1, day=1, tz="US/Pacific"), None),
        "string": value("a string", pd.StringDtype(storage="python")),
        # other types of features
        "class_label": other("positive", ClassLabel(names=["negative", "positive"])),
        "dict": other({"a": 0}, None),
        "list": other([{"a": 0}], None),
        "sequence_simple": other([0], None),
        "sequence": other([{"a": 0}], Sequence(feature={"a": Value(dtype="int64")})),
        "array2d": other(np.zeros((2, 2), dtype="float32"), Array2D(shape=(2, 2), dtype="float32")),
        "array3d": other(np.zeros((2, 2, 2), dtype="float32"), Array3D(shape=(2, 2, 2), dtype="float32")),
        "array4d": other(np.zeros((2, 2, 2, 2), dtype="float32"), Array4D(shape=(2, 2, 2, 2), dtype="float32")),
        "array5d": other(np.zeros((2, 2, 2, 2, 2), dtype="float32"), Array5D(shape=(2, 2, 2, 2, 2), dtype="float32")),
        "audio": other({"array": [0.1, 0.2, 0.3], "sampling_rate": sampling_rate}, Audio(sampling_rate=sampling_rate)),
        "image": other(str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"), Image()),
        "translation": other({"en": "the cat", "fr": "le chat"}, Translation(languages=["en", "fr"])),
        "translation_variable_languages": other(
            {"en": "the cat", "fr": ["le chat", "la chatte"]},
            TranslationVariableLanguages(languages=["en", "fr"]),
        ),
        "images_list": other(
            [
                str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
                str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
            ],
            [Image()],
        ),
        "audios_list": other(
            [
                {"array": [0.1, 0.2, 0.3], "sampling_rate": 16_000},
                {"array": [0.1, 0.2, 0.3], "sampling_rate": 16_000},
            ],
            [Audio()],
        ),
        "images_sequence": other(
            [
                str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
                str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
            ],
            Sequence(feature=Image()),
        ),
        "audios_sequence": other(
            [
                {"array": [0.1, 0.2, 0.3], "sampling_rate": 16_000},
                {"array": [0.1, 0.2, 0.3], "sampling_rate": 16_000},
            ],
            Sequence(feature=Audio()),
        ),
        "dict_of_audios_and_images": other(
            {
                "a": 0,
                "b": [
                    str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
                    str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"),
                ],
                "c": {
                    "ca": [
                        {"array": [0.1, 0.2, 0.3], "sampling_rate": 16_000},
                        {"array": [0.1, 0.2, 0.3], "sampling_rate": 16_000},
                    ]
                },
            },
            {"a": Value(dtype="int64"), "b": [Image()], "c": {"ca": [Audio()]}},
        ),
        "sequence_of_dicts": other(
            [{"a": {"b": 0}}, {"a": {"b": 1}}], Sequence(feature={"a": {"b": Value(dtype="int64")}})
        ),
        "none_value": other({"a": None}, {"a": Value(dtype="int64")}),
        "big": Dataset.from_pandas(
            pd.DataFrame({"col": ["a" * 1_234 for _ in range(4_567)]}, dtype=pd.StringDtype(storage="python"))
        ),
        "spawning_opt_in_out": Dataset.from_pandas(
            pd.DataFrame(
                {
                    "col": [
                        "http://testurl.test/test_image-optOut.jpg",
                        "http://testurl.test/test_image2.jpg",
                        "other",
                        "http://testurl.test/test_image3-optIn.png",
                    ]
                },
                dtype=pd.StringDtype(storage="python"),
            )
        ),
        "duckdb_index": Dataset.from_pandas(
            pd.DataFrame(
                {
                    "text": [
                        (
                            "Grand Moff Tarkin and Lord Vader are interrupted in their discussion by the buzz of the"
                            " comlink"
                        ),
                        "There goes another one.",
                        "Vader turns round and round in circles as his ship spins into space.",
                        "We count thirty Rebel ships, Lord Vader.",
                        "The wingman spots the pirateship coming at him and warns the Dark Lord",
                    ],
                    "column with spaces": [
                        "a",
                        "b",
                        "c",
                        "d",
                        "e",
                    ],
                },
                dtype=pd.StringDtype(storage="python"),
            )
        ),
        "descriptive_statistics": Dataset.from_dict(
            {
                "string_label__column": [
                    "cat",
                    "dog",
                    "cat",
                    "potato",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "potato",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "potato",
                    "dog",
                    "cat",
                    "dog",
                    "cat",
                ],
                "string_label__nan_column": [
                    "cat",
                    None,
                    "cat",
                    "potato",
                    "cat",
                    None,
                    "cat",
                    "cat",
                    "cat",
                    None,
                    "cat",
                    "cat",
                    "potato",
                    "cat",
                    "cat",
                    "cat",
                    "dog",
                    "cat",
                    None,
                    "cat",
                ],
                "int__column": [0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 5, 5, 5, 6, 7, 8, 8, 8],
                "int__nan_column": [0, None, 1, None, 2, None, 2, None, 4, None, 5, None, 5, 5, 5, 6, 7, 8, 8, 8],
                "float__column": [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    1.1,
                    2.2,
                    2.3,
                    2.6,
                    4.7,
                    5.1,
                    6.2,
                    6.7,
                    6.8,
                    7.0,
                    8.3,
                    8.4,
                    9.2,
                    9.7,
                    9.9,
                ],
                "float__nan_column": [
                    None,
                    0.2,
                    0.3,
                    None,
                    0.5,
                    None,
                    2.2,
                    None,
                    2.6,
                    4.7,
                    5.1,
                    None,
                    None,
                    None,
                    None,
                    8.3,
                    8.4,
                    9.2,
                    9.7,
                    9.9,
                ],
                "class_label__column": [
                    0,
                    1,
                    0,
                    0,
                    0,
                    -1,
                    0,
                    -1,
                    0,
                    -1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                ],
                "class_label__nan_column": [
                    0,
                    None,
                    0,
                    0,
                    0,
                    None,
                    -1,
                    0,
                    0,
                    None,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -1,
                    1,
                    0,
                    None,
                    0,
                ],
                "class_label__string_column": [
                    "cat",
                    "dog",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "dog",
                    "cat",
                    "dog",
                    "cat",
                ],
                "class_label__string_nan_column": [
                    "cat",
                    None,
                    "cat",
                    "cat",
                    "cat",
                    None,
                    "cat",
                    "cat",
                    "cat",
                    None,
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "dog",
                    "cat",
                    None,
                    "cat",
                ],
                "float__negative_column": [
                    -7.221,
                    -5.333,
                    -15.154,
                    -15.392,
                    -15.054,
                    -10.176,
                    -10.072,
                    -10.59,
                    -6.0,
                    -14.951,
                    -14.054,
                    -9.706,
                    -7.053,
                    -10.072,
                    -15.054,
                    -12.777,
                    -12.233,
                    -13.54,
                    -14.376,
                    -15.754,
                ],
                "float__cross_zero_column": [
                    -7.221,
                    -5.333,
                    -15.154,
                    -15.392,
                    -15.054,
                    -10.176,
                    -10.072,
                    -10.59,
                    6.0,
                    14.951,
                    14.054,
                    -9.706,
                    7.053,
                    0.0,
                    -15.054,
                    -12.777,
                    12.233,
                    13.54,
                    -14.376,
                    15.754,
                ],
                "float__large_values_column": [
                    1101.34567,
                    1178.923,
                    197.2134,
                    1150.8483,
                    169.907655,
                    156.4580,
                    134.4368456,
                    189.456,
                    145.0912,
                    148.354465,
                    190.8943,
                    1134.34,
                    155.22366,
                    153.0,
                    163.0,
                    143.5468,
                    177.231,
                    132.568,
                    191.99,
                    1114.0,
                ],
                "int__negative_column": [
                    -10,
                    -9,
                    -8,
                    -1,
                    -5,
                    -1,
                    -2,
                    -3,
                    -5,
                    -4,
                    -7,
                    -8,
                    -11,
                    -15,
                    -20 - 11,
                    -1,
                    -14,
                    -11,
                    -0,
                    -10,
                ],
                "int__cross_zero_column": [
                    -10,
                    -9,
                    -8,
                    0,
                    0,
                    1,
                    2,
                    -3,
                    -5,
                    4,
                    7,
                    8,
                    11,
                    15,
                    20 - 11,
                    -1,
                    14,
                    11,
                    0,
                    -10,
                ],
                "int__large_values_column": [
                    1101,
                    1178,
                    197,
                    1150,
                    169,
                    156,
                    134,
                    189,
                    145,
                    148,
                    190,
                    1134,
                    155,
                    153,
                    163,
                    143,
                    177,
                    132,
                    191,
                    1114,
                ],
            },
            features=Features(
                {
                    "string_label__column": Value("string"),
                    "string_label__nan_column": Value("string"),
                    "int__column": Value("int32"),
                    "int__nan_column": Value("int32"),
                    "int__negative_column": Value("int32"),
                    "int__cross_zero_column": Value("int32"),
                    "int__large_values_column": Value("int32"),
                    "float__column": Value("float32"),
                    "float__nan_column": Value("float32"),
                    "float__negative_column": Value("float64"),
                    "float__cross_zero_column": Value("float32"),
                    "float__large_values_column": Value("float32"),
                    "class_label__column": ClassLabel(names=["cat", "dog"]),
                    "class_label__nan_column": ClassLabel(names=["cat", "dog"]),
                    "class_label__string_column": ClassLabel(names=["cat", "dog"]),
                    "class_label__string_nan_column": ClassLabel(names=["cat", "dog"]),
                }
            ),
        ),
        "descriptive_statistics_string_text": Dataset.from_dict(
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
        ),
    }
