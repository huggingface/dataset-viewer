from libutils.enums import TimestampUnit


def test_timestamp_unit() -> None:
    assert TimestampUnit.s == TimestampUnit["s"]
    assert set(TimestampUnit.__members__) == {"s", "ms", "us", "ns"}
    assert {d.name for d in TimestampUnit} == {"s", "ms", "us", "ns"}
    assert "ms" in TimestampUnit.__members__
