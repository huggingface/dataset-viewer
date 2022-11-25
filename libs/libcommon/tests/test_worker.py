import pytest

from libcommon.worker import parse_version

from .utils import DummyWorker


@pytest.mark.parametrize(
    "string_version, expected_major_version, should_raise",
    [
        ("1.0.0", 1, False),
        ("3.1.2", 3, False),
        ("1.1", 1, False),
        ("not a version", None, True),
    ],
)
def test_parse_version(string_version: str, expected_major_version: int, should_raise: bool) -> None:
    if should_raise:
        with pytest.raises(Exception):
            parse_version(string_version)
    else:
        assert parse_version(string_version).major == expected_major_version


@pytest.mark.parametrize(
    "worker_version, other_version, expected, should_raise",
    [
        ("1.0.0", "1.0.1", 0, False),
        ("1.0.0", "2.0.1", -1, False),
        ("2.0.0", "1.0.1", 1, False),
        ("not a version", "1.0.1", None, True),
    ],
)
def test_compare_major_version(worker_version: str, other_version: str, expected: int, should_raise: bool) -> None:
    worker = DummyWorker(version=worker_version)
    if should_raise:
        with pytest.raises(Exception):
            worker.compare_major_version(other_version)
    else:
        assert worker.compare_major_version(other_version) == expected
