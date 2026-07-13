import fsspec
import pytest


def test_fsspec(tmpfs: fsspec.AbstractFileSystem) -> None:
    tmpfs.write_text("data.txt", "Hello, World!")
    with fsspec.open("tmp://data.txt", "r") as f:
        assert f.read() == "Hello, World!"
    with pytest.raises(ValueError):
        fsspec.open("simplecache::tmp://data.txt")
    with pytest.raises(ValueError):
        fsspec.open("data:,Hello%2C%20World%21")
