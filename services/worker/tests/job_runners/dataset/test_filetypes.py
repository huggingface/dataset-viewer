# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.


import pytest
from huggingface_hub.hf_api import RepoSibling

from worker.dtos import Filetype
from worker.job_runners.dataset.filetypes import get_filetypes, get_filetypes_from_archive


@pytest.mark.parametrize(
    "siblings,filetypes",
    [
        (
            [
                RepoSibling("file1.txt"),
                RepoSibling("file2.txt"),
            ],
            [
                Filetype(extension=".txt", count=2),
            ],
        ),
        (
            [
                RepoSibling("file1.txt"),
                RepoSibling("file2.txt"),
                RepoSibling("file3.csv"),
                RepoSibling("file3.tar"),
                RepoSibling("file3.tar.gz"),
                RepoSibling("file3.tar.gz_1"),
                RepoSibling("file3.tar.gz-1"),
                RepoSibling("file3.tar.gz?dl=1-1"),
                RepoSibling("file.gz"),
                RepoSibling(".gitignore"),
            ],
            [
                Filetype(extension=".txt", count=2),
                Filetype(extension=".csv", count=1),
                Filetype(extension=".tar", count=1),
                Filetype(extension=".tar.gz", count=4),
                Filetype(extension=".gz", count=1),
                Filetype(extension="", count=1),
            ],
        ),
    ],
)
def test_get_filetypes(siblings: list[RepoSibling], filetypes: list[Filetype]) -> None:
    assert get_filetypes(siblings) == filetypes


@pytest.mark.real_dataset
@pytest.mark.parametrize(
    "dataset,archive_filename,filetypes",
    [
        (
            "severo/LILA",
            "data/Caltech_Camera_Traps.jsonl.zip",
            [
                Filetype(extension=".jsonl", count=1, archived_in=".zip"),
            ],
        ),
        (
            "severo/winogavil",
            "winogavil_images.zip",
            [
                Filetype(extension=".jsonl", count=1, archived_in=".zip"),
            ],
        ),
    ],
)
def test_get_filetypes_from_archive(
    use_hub_prod_endpoint: pytest.MonkeyPatch,
    dataset: str,
    archive_filename: str,
    filetypes: list[Filetype],
) -> None:
    assert get_filetypes_from_archive(dataset=dataset, archive_filename=archive_filename, hf_token=None) == filetypes
