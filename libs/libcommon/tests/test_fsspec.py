import subprocess
import sys

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


@pytest.mark.parametrize("url", ["http://example.org/data.csv", "https://example.org/data.csv"])
def test_fsspec_refuses_http(url: str) -> None:
    with pytest.raises(ValueError, match="Protocol not known"):
        fsspec.core.url_to_fs(url)


# imports the HTTP filesystem before `libcommon`, i.e. before the protocols are removed
IMPORT_HTTP_FILESYSTEM_FIRST = """
import fsspec

fsspec.get_filesystem_class("https")

import libcommon  # noqa: F401

for url in ["http://example.org/data.csv", "https://example.org/data.csv"]:
    try:
        fsspec.core.url_to_fs(url)
    except ValueError:
        continue
    raise AssertionError(f"{url} is still allowed")
assert fsspec.get_filesystem_class("hf")
"""


def test_fsspec_refuses_http_whatever_the_import_order() -> None:
    # in a new process, since the import order is what is being checked
    subprocess.run([sys.executable, "-c", IMPORT_HTTP_FILESYSTEM_FIRST], check=True)
