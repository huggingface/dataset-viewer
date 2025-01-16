# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from datetime import datetime
from pathlib import Path
from typing import Optional

from datasets import Array2D, Audio, ClassLabel, Dataset, Features, Image, Sequence, Value

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


def long_text_null_column() -> list[Optional[str]]:
    texts = long_text_column()
    for i in range(0, len(texts), 7):
        texts[i] = None  # type: ignore
    return texts  # type: ignore


def null_column(n_samples: int) -> list[None]:
    return [None] * n_samples


statistics_dataset = Dataset.from_dict(
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
        "string_label__null_column": [
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
        "string_label__all_null_column": null_column(20),
        "int__column": [0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 5, 5, 5, 6, 7, 8, 8, 8],
        "int__null_column": [0, None, 1, None, 2, None, 2, None, 4, None, 5, None, 5, 5, 5, 6, 7, 8, 8, 8],
        "int__all_null_column": null_column(20),
        "int__only_one_value_column": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "int__only_one_value_null_column": [
            0,
            None,
            0,
            None,
            0,
            None,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
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
        "float__null_column": [
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
        "float__nan_column": [
            float("nan"),
            0.2,
            0.3,
            None,
            0.5,
            float("nan"),
            2.2,
            None,
            2.6,
            4.7,
            5.1,
            float("nan"),
            float("nan"),
            None,
            None,
            8.3,
            8.4,
            9.2,
            9.7,
            9.9,
        ],
        "float__all_null_column": null_column(20),
        "float__all_nan_column": [float("nan")] * 20,
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
        "class_label__less_classes_column": [
            0,
            0,
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
            0,
            0,
            0,
            0,
        ],
        "class_label__null_column": [
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
        "class_label__all_null_column": null_column(20),
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
        "class_label__string_null_column": [
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
        "class_label__string_all_null_column": null_column(20),
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
        "float__only_one_value_column": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "float__only_one_value_null_column": [
            0.0,
            None,
            0.0,
            None,
            0.0,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
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
        "bool__column": [
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
        ],
        "bool__null_column": [
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            None,
            None,
            None,
        ],
        "bool__all_null_column": null_column(20),
        "list__int_column": [
            [1],
            [1],
            [1],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [],
            [],
        ],
        "list__int_null_column": [
            [1],
            [1],
            [1],
            [None],
            [1, 2],
            [1, 2],
            [None],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [None],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [None],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [],
            [],
        ],
        "list__int_all_null_column": null_column(20),
        "list__string_column": [
            ["cat"],
            ["cat"],
            ["cat"],
            ["cat", "sat"],
            ["cat", "sat"],
            ["cat", "sat"],
            ["cat", "sat", "mat"],
            ["cat", "sat", "mat"],
            ["cat", "sat", "mat"],
            ["cat", "sat", "mat"],
            ["cat", "sat", "mat"],
            ["cat", "sat", "mat", "blat"],
            ["cat", "sat", "mat", "blat", "flat", "hat", "none of that", None],
            ["cat", "sat", "mat", "blat", "flat", "hat", "none of that", None, "this is sad..."],
            ["cat", "sat", "mat", "blat", "flat", "hat", "none of that", None, "this is sad..."],
            ["cat", "sat", "mat", "blat", "flat", "hat", "none of that", None, "this is sad..."],
            ["cat", "sat", "mat", "blat", "flat", "hat", "none of that", None, "this is sad..."],
            ["cat", "sat", "mat", "blat", "flat", "hat", "none of that", None, "this is sad..."],
            [],
            [],
        ],
        "list__string_null_column": [
            ["cat"],
            ["cat"],
            ["cat"],
            [None],
            ["cat", "sat"],
            ["cat", "sat"],
            [None],
            ["cat", "sat", "mat"],
            ["cat", "sat", "mat"],
            ["cat", "sat", "mat"],
            ["cat", "sat", "mat"],
            [None],
            ["cat", "sat", "mat", "blat", "flat", "hat", "none of that", None],
            ["cat", "sat", "mat", "blat", "flat", "hat", "none of that", None, "this is sad..."],
            ["cat", "sat", "mat", "blat", "flat", "hat", "none of that", None, "this is sad..."],
            [None],
            ["cat", "sat", "mat", "blat", "flat", "hat", "none of that", None, "this is sad..."],
            ["cat", "sat", "mat", "blat", "flat", "hat", "none of that", None, "this is sad..."],
            [],
            [],
        ],
        "list__string_all_null_column": null_column(20),
        "list__dict_column": [
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
        "list__dict_null_column": [
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
        "list__dict_all_null_column": null_column(20),
        "list__sequence_int_column": [
            [1],
            [1],
            [1],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [],
            [],
        ],
        "list__sequence_int_null_column": [
            [1],
            [1],
            [1],
            None,
            [1, 2],
            [1, 2],
            None,
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            None,
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            None,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [],
            [],
        ],
        "list__sequence_int_all_null_column": null_column(20),
        "list__sequence_class_label_column": [
            ["cat"],
            ["cat"],
            ["cat"],
            ["cat", "dog"],
            ["cat", "cat"],
            ["cat", "cat"],
            ["cat", "cat"],
            ["cat", "cat"],
            ["cat", "cat", "cat", "dog"],
            ["cat", "cat", "cat", "dog"],
            ["cat", "cat", "cat", "dog", "cat", "cat", "dog"],
            ["cat", "cat", "cat", "dog", "cat", "cat", "dog"],
            ["cat", "cat", "cat", "dog", "cat", "cat", "dog"],
            ["cat", "cat", "cat", "dog", "cat", "cat", "dog"],
            ["cat", "cat", "cat", "dog", "cat", "cat", "dog"],
            ["cat", "cat", "cat", "dog", "cat", "cat", "dog"],
            [],
            [],
            [],
            [],
        ],
        "list__sequence_class_label_null_column": [
            ["cat"],
            ["cat"],
            None,
            ["cat", "dog"],
            ["cat", "cat"],
            ["cat", "cat"],
            ["cat", "cat"],
            None,
            ["cat", "cat", "cat", "dog"],
            ["cat", "cat", "cat", "dog"],
            ["cat", "cat", "cat", "dog", "cat", "cat", "dog"],
            ["cat", "cat", "cat", "dog", "cat", "cat", "dog"],
            None,
            None,
            ["cat", "cat", "cat", "dog", "cat", "cat", "dog"],
            ["cat", "cat", "cat", "dog", "cat", "cat", "dog"],
            [],
            [],
            [],
            [],
        ],
        "list__sequence_class_label_all_null_column": null_column(20),
        "list__sequence_of_sequence_bool_column": [
            [[True]],
            [[True]],
            [[True]],
            [[True], [None, True, False], []],
            [[True, False]],
            [[True, False], [True, False], [True]],
            [[True]],
            [[True, False, False], [True, False], [], [True, False], [True, False, True, False]],
            [[True, False, False]],
            [[True, False, False]],
            [[True, False, False]],
            [[True, False, False, False], [True], [], [None, True]],
            [[True, False, False, False], [True], [None, True]],
            [[True, False, False, False, True], [None, True]],
            [[True], [False], [False], [False, None], [True, None, True]],
            [[True]],
            [[True, False], [False, False, True], [None, True]],
            [[True, False, False, False], [None, True]],
            [[True], [None, True, False]],
            [[True], [True, False], [True, False]],
        ],
        "list__sequence_of_sequence_bool_null_column": [
            [[True]],
            [[True]],
            [[True]],
            [[True], [None, True, False]],
            None,
            [[True, False], [True, False], [True], []],
            [[True]],
            [[True, False, False], [True, False], [True, False], [], [True, False, True, False]],
            [[True, False, False]],
            None,
            [[True, False, False]],
            [[True, False, False, False], [True], [None, True]],
            [[True, False, False, False], [True], [None, True]],
            [[True, False, False, False, True], [None, True]],
            [[True], [False], [False], [False, None], [True, None, True]],
            [[True]],
            None,
            [[True, False, False, False], [None, True]],
            [[True], [None, True, False]],
            [[True], [True, False], [True, False]],
        ],
        "list__sequence_of_sequence_bool_all_null_column": null_column(20),
        "list__sequence_of_sequence_dict_column": [
            [[{"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}, {}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}, {}]],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [[]],
            [[]],
        ],
        "list__sequence_of_sequence_dict_null_column": [
            [[{"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {}, {"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}, {}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}, {}]],
            None,
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            None,
            None,
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ],
                [],
                [],
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [[]],
            [[]],
        ],
        "list__sequence_of_sequence_dict_all_null_column": null_column(20),
        "list__sequence_of_list_dict_column": [
            [[{"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}, {}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}, {}]],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [[]],
            [[]],
        ],
        "list__sequence_of_list_dict_null_column": [
            [[{"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {}, {"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}, {}]],
            [[{"author": "cat", "likes": 5}, {"author": "cat", "likes": 5}, {}]],
            None,
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            None,
            None,
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ],
                [],
                [],
            ],
            [
                [
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                    {"author": "cat", "likes": 5},
                ]
            ],
            [[]],
            [[]],
        ],
        "list__sequence_of_list_dict_all_null_column": null_column(20),
        "array__list_column": [
            [[[1, 2, 3]]],
            [[[1, 2, 3]]],
            [[[1, 2, 3]]],
            [[[4, 5, 6]], [[4, 5, 6]]],
            [[[4, 5, 6]], [[4, 5, 6]]],
            [[[4, 5, 6]], [[4, 5, 6]]],
            [[[7, 8, 9]]],
            [[[7, 8, 9]]],
            [[[7, 8, 9]], [[7, 8, 9]], [[7, 8, 9]]],
            [[[10, 11, 12]]],
            [[[10, 11, 12]]],
            [[[10, 11, 12]]],
            [[[13, 14, 15]]],
            [[[13, 14, 15]]],
            [[[13, 14, 15]]],
            [[[16, 17, 18]]],
            [[[16, 17, 18]]],
            [[[16, 17, 18]]],
            [[[19, 20, 21]]],
            [[[19, 20, 21]], [[19, 20, 21]], [[19, 20, 21]], [[19, 20, 21]]],
        ],
        "array__sequence_column": [
            [[[1, 2, 3]]],
            [[[1, 2, 3]]],
            [[[1, 2, 3]]],
            [[[4, 5, 6]], [[4, 5, 6]]],
            [[[4, 5, 6]], [[4, 5, 6]]],
            [[[4, 5, 6]], [[4, 5, 6]]],
            [[[7, 8, 9]]],
            [[[7, 8, 9]]],
            [[[7, 8, 9]], [[7, 8, 9]], [[7, 8, 9]]],
            [[[10, 11, 12]]],
            [[[10, 11, 12]]],
            [[[10, 11, 12]]],
            [[[13, 14, 15]]],
            [[[13, 14, 15]]],
            [[[13, 14, 15]]],
            [[[16, 17, 18]]],
            [[[16, 17, 18]]],
            [[[16, 17, 18]]],
            [[[19, 20, 21]]],
            [[[19, 20, 21]], [[19, 20, 21]], [[19, 20, 21]], [[19, 20, 21]]],
        ],
    },
    features=Features(
        {
            "string_label__column": Value("string"),
            "string_label__null_column": Value("string"),
            "string_label__all_null_column": Value("string"),
            "int__column": Value("int32"),
            "int__null_column": Value("int32"),
            "int__all_null_column": Value("int32"),
            "int__negative_column": Value("int32"),
            "int__cross_zero_column": Value("int32"),
            "int__large_values_column": Value("int32"),
            "int__only_one_value_column": Value("int32"),
            "int__only_one_value_null_column": Value("int32"),
            "float__column": Value("float32"),
            "float__null_column": Value("float32"),
            "float__nan_column": Value("float32"),
            "float__all_null_column": Value("float32"),
            "float__all_nan_column": Value("float32"),
            "float__negative_column": Value("float64"),
            "float__cross_zero_column": Value("float32"),
            "float__large_values_column": Value("float32"),
            "float__only_one_value_column": Value("float32"),
            "float__only_one_value_null_column": Value("float32"),
            "class_label__column": ClassLabel(names=["cat", "dog"]),
            "class_label__null_column": ClassLabel(names=["cat", "dog"]),
            "class_label__all_null_column": ClassLabel(names=["cat", "dog"]),
            "class_label__less_classes_column": ClassLabel(names=["cat", "dog"]),  # but only "cat" is in set
            "class_label__string_column": ClassLabel(names=["cat", "dog"]),
            "class_label__string_null_column": ClassLabel(names=["cat", "dog"]),
            "class_label__string_all_null_column": ClassLabel(names=["cat", "dog"]),
            "bool__column": Value("bool"),
            "bool__null_column": Value("bool"),
            "bool__all_null_column": Value("bool"),
            "list__int_column": [Value("int32")],
            "list__int_null_column": [Value("int32")],
            "list__int_all_null_column": [Value("int32")],
            "list__string_column": [Value("string")],
            "list__string_null_column": [Value("string")],
            "list__string_all_null_column": [Value("string")],
            "list__dict_column": [{"author": Value("string"), "content": Value("string"), "likes": Value("int32")}],
            "list__dict_null_column": [
                {"author": Value("string"), "content": Value("string"), "likes": Value("int32")}
            ],
            "list__dict_all_null_column": [
                {"author": Value("string"), "content": Value("string"), "likes": Value("int32")}
            ],
            "list__sequence_int_column": Sequence(Value("int64")),
            "list__sequence_int_null_column": Sequence(Value("int64")),
            "list__sequence_int_all_null_column": Sequence(Value("int64")),
            "list__sequence_class_label_column": Sequence(ClassLabel(names=["cat", "dog"])),
            "list__sequence_class_label_null_column": Sequence(ClassLabel(names=["cat", "dog"])),
            "list__sequence_class_label_all_null_column": Sequence(ClassLabel(names=["cat", "dog"])),
            "list__sequence_of_sequence_bool_column": Sequence(Sequence(Value("bool"))),
            "list__sequence_of_sequence_bool_null_column": Sequence(Sequence(Value("bool"))),
            "list__sequence_of_sequence_bool_all_null_column": Sequence(Sequence(Value("bool"))),
            "list__sequence_of_sequence_dict_column": Sequence(
                Sequence({"author": Value("string"), "likes": Value("int32")})
            ),
            "list__sequence_of_sequence_dict_null_column": Sequence(
                Sequence({"author": Value("string"), "likes": Value("int32")})
            ),
            "list__sequence_of_sequence_dict_all_null_column": Sequence(
                Sequence({"author": Value("string"), "likes": Value("int32")})
            ),
            "list__sequence_of_list_dict_column": Sequence([{"author": Value("string"), "likes": Value("int32")}]),
            "list__sequence_of_list_dict_null_column": Sequence(
                [{"author": Value("string"), "likes": Value("int32")}]
            ),
            "list__sequence_of_list_dict_all_null_column": Sequence(
                [{"author": Value("string"), "likes": Value("int32")}]
            ),
            "array__list_column": [Array2D(shape=(1, 3), dtype="int32")],
            "array__sequence_column": Sequence(Array2D(shape=(1, 3), dtype="int32")),
        }
    ),
)

statistics_string_text_dataset = Dataset.from_dict(
    {
        "string_text__column": long_text_column(),
        "string_text__null_column": long_text_null_column(),
        "string_text__large_string_column": long_text_column(),
        "string_text__large_string_null_column": long_text_null_column(),
    },
    features=Features(
        {
            "string_text__column": Value("string"),
            "string_text__null_column": Value("string"),
            "string_text__large_string_column": Value("large_string"),
            "string_text__large_string_null_column": Value("large_string"),
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
        "list__sequence_dict_null_column": [
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
        "list__sequence_dict_all_null_column": null_column(20),
    },
    features=Features(
        {
            "list__sequence_dict_column": Sequence(
                {"author": Value("string"), "content": Value("string"), "likes": Value("int32")}
            ),
            "list__sequence_dict_null_column": Sequence(
                {"author": Value("string"), "content": Value("string"), "likes": Value("int32")}
            ),
            "list__sequence_dict_all_null_column": Sequence(
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
        "audio_null": [
            str(Path(__file__).resolve().parent / "data" / "audio" / "audio_1.wav"),
            None,
            str(Path(__file__).resolve().parent / "data" / "audio" / "audio_3.wav"),
            None,
        ],
        "audio_all_null": [None, None, None, None],
    },
    features=Features(
        {
            "audio": Audio(sampling_rate=1600, decode=False),
            "audio_null": Audio(sampling_rate=1600, decode=False),
            "audio_all_null": Audio(sampling_rate=1600, decode=False),
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
        "image_null": [
            str(Path(__file__).resolve().parent / "data" / "image" / "image_1.jpg"),
            None,
            str(Path(__file__).resolve().parent / "data" / "image" / "image_3.jpg"),
            None,
        ],
        "image_all_null": [None, None, None, None],
    },
    features=Features(
        {
            "image": Image(decode=False),
            "image_null": Image(decode=False),
            "image_all_null": Image(decode=False),
        }
    ),
)


datetime_dataset = Dataset.from_dict(
    {
        "datetime_string": [
            "2024-01-01 00:00:00",
            "2024-01-02 00:00:00",
            "2024-01-03 00:00:00",
            "2024-01-04 00:00:00",
            "2024-01-05 00:00:00",
            "2024-01-06 00:00:00",
            "2024-01-07 00:00:00",
            "2024-01-08 00:00:00",
            "2024-01-09 00:00:00",
            "2024-01-10 00:00:00",
            "2024-01-11 00:00:00",
        ],
        "datetime_string_z": [
            "2024-01-01 00:00:00Z",
            "2024-01-02 00:00:00Z",
            "2024-01-03 00:00:00Z",
            "2024-01-04 00:00:00Z",
            "2024-01-05 00:00:00Z",
            "2024-01-06 00:00:00Z",
            "2024-01-07 00:00:00Z",
            "2024-01-08 00:00:00Z",
            "2024-01-09 00:00:00Z",
            "2024-01-10 00:00:00Z",
            "2024-01-11 00:00:00Z",
        ],
        "datetime_string_t_z": [
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z",
            "2024-01-03T00:00:00Z",
            "2024-01-04T00:00:00Z",
            "2024-01-05T00:00:00Z",
            "2024-01-06T00:00:00Z",
            "2024-01-07T00:00:00Z",
            "2024-01-08T00:00:00Z",
            "2024-01-09T00:00:00Z",
            "2024-01-10T00:00:00Z",
            "2024-01-11T00:00:00Z",
        ],
        "datetime_string_tz": [
            "2024-01-01 00:00:00+0200",
            "2024-01-02 00:00:00+0200",
            "2024-01-03 00:00:00+0200",
            "2024-01-04 00:00:00+0200",
            "2024-01-05 00:00:00+0200",
            "2024-01-06 00:00:00+0200",
            "2024-01-07 00:00:00+0200",
            "2024-01-08 00:00:00+0200",
            "2024-01-09 00:00:00+0200",
            "2024-01-10 00:00:00+0200",
            "2024-01-11 00:00:00+0200",
        ],
        "datetime_string_error": [
            "16/01/2023",
            "17/01/2023",
            "18/01/2023",
            "19/01/2023",
            "01/2023",
            "02/2023",
            "20/01/2023",
            "21/01/2023",
            "03/2023",
            "25/01/2023",
            "26/01/2023",
        ],
        "datetime": [
            datetime.strptime("2024-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2024-01-02 00:00:00", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2024-01-03 00:00:00", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2024-01-04 00:00:00", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2024-01-05 00:00:00", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2024-01-06 00:00:00", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2024-01-07 00:00:00", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2024-01-08 00:00:00", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2024-01-09 00:00:00", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2024-01-10 00:00:00", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2024-01-11 00:00:00", "%Y-%m-%d %H:%M:%S"),
        ],
        "datetime_tz": [
            datetime.strptime("2024-01-01 00:00:00+0200", "%Y-%m-%d %H:%M:%S%z"),
            datetime.strptime("2024-01-02 00:00:00+0200", "%Y-%m-%d %H:%M:%S%z"),
            datetime.strptime("2024-01-03 00:00:00+0200", "%Y-%m-%d %H:%M:%S%z"),
            datetime.strptime("2024-01-04 00:00:00+0200", "%Y-%m-%d %H:%M:%S%z"),
            datetime.strptime("2024-01-05 00:00:00+0200", "%Y-%m-%d %H:%M:%S%z"),
            datetime.strptime("2024-01-06 00:00:00+0200", "%Y-%m-%d %H:%M:%S%z"),
            datetime.strptime("2024-01-07 00:00:00+0200", "%Y-%m-%d %H:%M:%S%z"),
            datetime.strptime("2024-01-08 00:00:00+0200", "%Y-%m-%d %H:%M:%S%z"),
            datetime.strptime("2024-01-09 00:00:00+0200", "%Y-%m-%d %H:%M:%S%z"),
            datetime.strptime("2024-01-10 00:00:00+0200", "%Y-%m-%d %H:%M:%S%z"),
            datetime.strptime("2024-01-11 00:00:00+0200", "%Y-%m-%d %H:%M:%S%z"),
        ],
        "datetime_null": [
            datetime.strptime("2024-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
            None,
            datetime.strptime("2024-01-03 00:00:00", "%Y-%m-%d %H:%M:%S"),
            None,
            datetime.strptime("2024-01-05 00:00:00", "%Y-%m-%d %H:%M:%S"),
            None,
            datetime.strptime("2024-01-07 00:00:00", "%Y-%m-%d %H:%M:%S"),
            None,
            datetime.strptime("2024-01-09 00:00:00", "%Y-%m-%d %H:%M:%S"),
            None,
            datetime.strptime("2024-01-11 00:00:00", "%Y-%m-%d %H:%M:%S"),
        ],
        "datetime_all_null": [None] * 11,
    },
    features=Features(
        {
            "datetime_string": Value("string"),
            "datetime_string_z": Value("string"),
            "datetime_string_t_z": Value("string"),
            "datetime_string_tz": Value("string"),
            "datetime_string_error": Value("string"),
            "datetime": Value("timestamp[s]"),
            "datetime_tz": Value("timestamp[s, tz=+02:00]"),
            "datetime_null": Value("timestamp[s]"),
            "datetime_all_null": Value("timestamp[s]"),
        }
    ),
)
