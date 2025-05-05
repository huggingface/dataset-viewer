from typing import Optional

import pytest
from pytest import TempPathFactory

from libapi.duckdb import check_available_disk_space

# TODO(QL): test duckdb indexing


@pytest.mark.parametrize("subpath", [None, "does_not_exist"])
def test_check_available_disk_space(tmp_path_factory: TempPathFactory, subpath: Optional[str]) -> None:
    path = tmp_path_factory.mktemp("test_check_available_disk_space")
    if subpath:
        path = path / subpath
    check_available_disk_space(path=path, required_space=1)
