from enum import Enum, auto

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


class LabelsColumnType(Enum):
    CLASS_LABEL = auto()


class TimestampColumnType(Enum):
    TIMESTAMP = auto()
