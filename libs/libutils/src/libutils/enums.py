from enum import Enum, auto
from typing import Literal, Union

# in the enums defined here, we only use the names, and never the values.
# see the test (tests/test_enums.py) for how to manage them:
# MiEnum.__members__ or MiEnum.MiMember.name


class TimestampUnit(Enum):
    s = auto()
    ms = auto()
    us = auto()
    ns = auto()


class CommonColumnType(Enum):
    JSON = auto()
    BOOL = auto()
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    IMAGE_URL = auto()
    RELATIVE_IMAGE_URL = auto()
    AUDIO_RELATIVE_SOURCES = auto()


# TODO: generate CommonColumnTypeName from CommonColumnType?
CommonColumnTypeName = Union[
    Literal["JSON"],
    Literal["BOOL"],
    Literal["INT"],
    Literal["FLOAT"],
    Literal["STRING"],
    Literal["IMAGE_URL"],
    Literal["RELATIVE_IMAGE_URL"],
    Literal["AUDIO_RELATIVE_SOURCES"],
]


class LabelsColumnType(Enum):
    CLASS_LABEL = auto()


# TODO: generate LabelsColumnTypeName from LabelsColumnType?
LabelsColumnTypeName = Literal["CLASS_LABEL"]


class TimestampColumnType(Enum):
    TIMESTAMP = auto()


# TODO: generate TimestampColumnTypeName from TimestampColumnType?
TimestampColumnTypeName = Literal["TIMESTAMP"]
