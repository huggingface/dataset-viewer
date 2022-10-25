import pytest

from libqueue.worker import compare_major_version, parse_version

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
    "semver_a, semver_b, expected, should_raise",
    [
        ("1.0.0", "1.0.1", 0, False),
        ("1.0.0", "2.0.1", -1, False),
        ("2.0.0", "1.0.1", 1, False),
        ("not a version", "1.0.1", None, True),
    ],
)
def test_compare_major_version(semver_a: str, semver_b: str, expected: int, should_raise: bool) -> None:
    if should_raise:
        with pytest.raises(Exception):
            compare_major_version(semver_a, semver_b)
    else:
        assert compare_major_version(semver_a, semver_b) == expected


@pytest.mark.parametrize(
    "old_version, worker_version, expected, should_raise",
    [
        ("1.0.0", "1.0.1", False, False),
        ("1.0.0", "2.0.1", True, False),
        ("2.0.0", "1.0.1", False, False),
        ("not a version", "1.0.1", None, True),
    ],
)
def test_is_major_version_lower_than_worker(
    old_version: str, worker_version: str, expected: int, should_raise: bool
) -> None:
    worker = DummyWorker(version=worker_version)
    if should_raise:
        with pytest.raises(Exception):
            worker.is_major_version_lower_than_worker(old_version)
    else:
        assert worker.is_major_version_lower_than_worker(old_version) == expected
