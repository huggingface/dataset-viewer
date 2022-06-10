from enum import Enum, auto


# in the enums defined here, we only use the names, and never the values.
# see the test (tests/test_enums.py) for how to manage them:
# MiEnum.__members__ or MiEnum.MiMember.name


class TimestampUnit(Enum):
    s = auto()
    ms = auto()
    us = auto()
    ns = auto()
