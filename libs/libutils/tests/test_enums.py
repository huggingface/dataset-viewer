from libutils.enums import (
    CommonColumnType,
    LabelsColumnType,
    TimestampColumnType,
    TimestampUnit,
)


def test_timestamp_unit() -> None:
    assert TimestampUnit.s == TimestampUnit["s"]
    assert set(TimestampUnit.__members__) == {"s", "ms", "us", "ns"}
    assert {d.name for d in TimestampUnit} == {"s", "ms", "us", "ns"}
    assert "ms" in TimestampUnit.__members__


def test_column_type() -> None:
    assert set(CommonColumnType.__members__) == {
        "JSON",
        "BOOL",
        "INT",
        "FLOAT",
        "STRING",
        "IMAGE_URL",
        "RELATIVE_IMAGE_URL",
        "AUDIO_RELATIVE_SOURCES",
    }
    assert set(LabelsColumnType.__members__) == {"CLASS_LABEL"}
    assert set(TimestampColumnType.__members__) == {"TIMESTAMP"}
