from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest
from pytest import TempPathFactory

from libapi.duckdb import get_index_file_location_and_download_if_missing


@pytest.mark.parametrize("partial_index", [False, True])
@pytest.mark.parametrize("partial_split", [False, True])
async def test_get_index_file_location_and_download_if_missing(
    partial_split: bool, partial_index: bool, tmp_path_factory: TempPathFactory
) -> None:
    duckdb_index_file_directory = (
        tmp_path_factory.mktemp("test_get_index_file_location_and_download_if_missing") / "duckdb"
    )
    duckdb_index_file_directory.mkdir(parents=True, exist_ok=True)

    dataset = "dataset"
    revision = "revision"
    target_revision = "refs/convert/parquet"
    config = "config"
    split = "split"
    default_filename = "index.duckdb"
    split_directory = f"partial-{split}" if partial_split else split
    filename = f"partial-{default_filename}" if partial_index else default_filename
    url = f"https://foo.bar/{dataset}/{target_revision}/resolve/{config}/{split_directory}/{filename}"

    def download_index_file(
        cache_folder: str,
        index_folder: str,
        target_revision: str,
        dataset: str,
        repo_file_location: str,
        hf_token: Optional[str] = None,
    ) -> None:
        Path(index_folder, repo_file_location).parent.mkdir(parents=True, exist_ok=True)
        Path(index_folder, repo_file_location).touch()

    expected_repo_file_location = f"{config}/{split_directory}/{filename}"
    with patch("libapi.duckdb.download_index_file", side_effect=download_index_file) as download_mock:
        await get_index_file_location_and_download_if_missing(
            duckdb_index_file_directory=duckdb_index_file_directory,
            dataset=dataset,
            config=config,
            split=split,
            revision=revision,
            filename=filename,
            url=url,
            target_revision=target_revision,
            hf_token=None,
        )
        download_mock.assert_called_once()
        args, kwargs = download_mock.call_args
        assert not args
        assert kwargs["repo_file_location"] == expected_repo_file_location
