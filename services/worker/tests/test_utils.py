# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from contextlib import AbstractContextManager
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import patch

import pytest
from datasets.packaged_modules.csv.csv import Csv
from huggingface_hub.hf_api import GitCommitInfo, GitRefInfo, GitRefs, HfApi

from worker.utils import (
    DATASET_TYPE,
    FileExtension,
    create_branch,
    get_file_extension,
    get_initial_commit,
    safe_load_dataset_builder,
)


@pytest.mark.parametrize(
    "filename,expected_extension",
    [
        ("README.md", FileExtension(extension=".md")),
        ("file.csv", FileExtension(extension=".csv")),
        # leading dots are ignored
        (".gitattributes", FileExtension(extension="")),
        (".file.csv", FileExtension(extension=".csv")),
        ("....file.csv", FileExtension(extension=".csv")),
        # no extension
        ("LICENSE", FileExtension(extension="")),
        # multiple dots
        ("file.with.dots.csv", FileExtension(extension=".csv")),
        # clean suffixes
        ("file.csv?dl=1", FileExtension(extension=".csv")),
        ("file.csv_1", FileExtension(extension=".csv")),
        ("file.csv-00000-of-00001", FileExtension(extension=".csv")),
        # ignore paths
        ("path/to/file.csv", FileExtension(extension=".csv")),
        (".path/to.some/file.csv", FileExtension(extension=".csv")),
        ("path/to/.gitignore", FileExtension(extension="")),
        # double extensions
        ("file.tar.gz", FileExtension(extension=".gz", uncompressed_extension=".tar")),
        ("file.with.dots.tar.gz", FileExtension(extension=".gz", uncompressed_extension=".tar")),
        ("file.tar.bz2", FileExtension(extension=".bz2", uncompressed_extension=".tar")),
        ("file.jsonl.gz", FileExtension(extension=".gz", uncompressed_extension=".jsonl")),
        ("file.tar.unknown", FileExtension(extension=".unknown")),
        ("file.tar", FileExtension(extension=".tar")),
        # case insensitive
        ("file.CSV", FileExtension(extension=".csv")),
        ("file.CSv", FileExtension(extension=".csv")),
        ("file.CSV?dl=1", FileExtension(extension=".csv")),
        ("file.with.dots.TAR.GZ", FileExtension(extension=".gz", uncompressed_extension=".tar")),
    ],
)
def test_get_file_extension(filename: str, expected_extension: FileExtension) -> None:
    assert get_file_extension(filename).extension == expected_extension.extension
    assert get_file_extension(filename).uncompressed_extension == expected_extension.uncompressed_extension


def _get_dataset_module() -> SimpleNamespace:
    return SimpleNamespace(
        builder_configs_parameters=SimpleNamespace(default_config_name="default"),
        builder_kwargs={"base_path": "hf://datasets/namespace/dataset@revision"},
        dataset_infos={},
        hash="revision",
    )


def test_safe_load_dataset_builder_allows_non_arrow_builder() -> None:
    class CsvBuilder(Csv):  # type: ignore[misc]
        builder_configs = {"default": SimpleNamespace(data_files=None)}

        def __init__(self, **kwargs: object) -> None:
            pass

    with (
        patch("datasets.load.dataset_module_factory", return_value=_get_dataset_module()),
        patch("datasets.load.get_dataset_builder_class", return_value=CsvBuilder),
    ):
        # data_files=None and download_mode=None mirror what `datasets.inspect` passes
        builder = safe_load_dataset_builder(
            path="namespace/dataset", name="default", data_files=None, download_mode=None
        )

    assert isinstance(builder, CsvBuilder)


class FakeResponse:
    """Minimal stand-in for the http response returned by `http_backoff`."""

    def __init__(self, headers: dict[str, str], json_data: Any = None) -> None:
        self.headers = headers
        self._json_data = json_data

    def json(self) -> Any:
        return self._json_data

    def raise_for_status(self) -> None:
        return None


class ListCommitsHfApi(HfApi):
    """HfApi that records the calls to `list_repo_commits` (the slow path we want to avoid)."""

    def __init__(self) -> None:
        super().__init__(endpoint="https://huggingface.co", token="api_token")
        self.listed_revisions: list[str] = []

    def list_repo_commits(self, *args: Any, **kwargs: Any) -> list[GitCommitInfo]:
        self.listed_revisions.append(str(kwargs.get("revision")))
        return [
            GitCommitInfo(
                commit_id="last-sha",
                authors=["user"],
                created_at=datetime(2024, 1, 2),
                title="last commit",
                message="",
                formatted_title=None,
                formatted_message=None,
            ),
            GitCommitInfo(
                commit_id="root-sha",
                authors=["system"],
                created_at=datetime(2024, 1, 1),
                title="initial commit",
                message="",
                formatted_title=None,
                formatted_message=None,
            ),
        ]


def _patch_http_backoff(
    responses: dict[str, FakeResponse], calls: list[tuple[str, str, Any]]
) -> AbstractContextManager[Any]:
    def _http_backoff(method: str, url: str, **kwargs: Any) -> FakeResponse:
        calls.append((method, url, kwargs.get("params")))
        return responses[method]

    return patch("worker.utils.http_backoff", side_effect=_http_backoff)


def test_get_initial_commit_asks_only_two_pages() -> None:
    calls: list[tuple[str, str, Any]] = []
    responses = {
        "HEAD": FakeResponse(headers={"X-Total-Count": "18"}),
        "GET": FakeResponse(headers={}, json_data=[{"id": "root-sha"}]),
    }
    hf_api = ListCommitsHfApi()
    with _patch_http_backoff(responses, calls):
        commit = get_initial_commit(hf_api, "user/dataset")
    assert commit == "root-sha"
    # the count is fetched with a HEAD request, then only the last page (index count - 1) is requested
    assert calls == [
        (
            "HEAD",
            "https://huggingface.co/api/datasets/user/dataset/commits/main",
            {"limit": 1},
        ),
        (
            "GET",
            "https://huggingface.co/api/datasets/user/dataset/commits/main",
            {"limit": 1, "p": 17},
        ),
    ]
    assert hf_api.listed_revisions == []


def test_get_initial_commit_on_a_pinned_revision() -> None:
    calls: list[tuple[str, str, Any]] = []
    responses = {
        "HEAD": FakeResponse(headers={"X-Total-Count": "1"}),
        "GET": FakeResponse(headers={}, json_data=[{"id": "root-sha"}]),
    }
    with _patch_http_backoff(responses, calls):
        commit = get_initial_commit(ListCommitsHfApi(), "user/dataset", revision="a" * 40)
    assert commit == "root-sha"
    assert [call[2] for call in calls] == [{"limit": 1}, {"limit": 1, "p": 0}]
    # a revision with a single commit is its own initial commit, and the requests are made on the pinned revision
    assert all(call[1] == "https://huggingface.co/api/datasets/user/dataset/commits/" + "a" * 40 for call in calls)


@pytest.mark.parametrize(
    "head_response,get_response",
    [
        (FakeResponse(headers={}), FakeResponse(headers={}, json_data=[{"id": "root-sha"}])),  # no count
        (FakeResponse(headers={"X-Total-Count": "3"}), FakeResponse(headers={}, json_data=[])),  # empty last page
    ],
)
def test_get_initial_commit_falls_back_on_listing_the_commits(
    head_response: FakeResponse, get_response: FakeResponse
) -> None:
    calls: list[tuple[str, str, Any]] = []
    hf_api = ListCommitsHfApi()
    with _patch_http_backoff({"HEAD": head_response, "GET": get_response}, calls):
        commit = get_initial_commit(hf_api, "user/dataset", revision="refs/convert/parquet")
    assert commit == "root-sha"
    # the revision is quoted in the url, and passed as-is to list_repo_commits
    assert all(call[1].endswith("/commits/refs%2Fconvert%2Fparquet") for call in calls)
    assert hf_api.listed_revisions == ["refs/convert/parquet"]


@pytest.mark.real_dataset
@pytest.mark.parametrize("dataset", ["abisee/cnn_dailymail", "lhoestq/test"])
def test_get_initial_commit_from_hub(use_hub_prod_endpoint: pytest.MonkeyPatch, dataset: str) -> None:
    # ^ the fast path must return the same commit as the full list of commits, whose last item is the root commit
    hf_api = HfApi(endpoint="https://huggingface.co")
    initial_commit = get_initial_commit(hf_api, dataset, repo_type=DATASET_TYPE)
    assert initial_commit == hf_api.list_repo_commits(repo_id=dataset, repo_type=DATASET_TYPE)[-1].commit_id


class RefsHfApi(HfApi):
    """HfApi with fake refs, that records the branches it has been asked to create."""

    def __init__(self, converts: Optional[list[GitRefInfo]] = None) -> None:
        super().__init__(endpoint="https://huggingface.co", token="api_token")
        self.converts = converts or []
        self.created_branches: list[tuple[str, str]] = []

    def list_repo_refs(self, *args: Any, **kwargs: Any) -> GitRefs:
        return GitRefs(
            branches=[GitRefInfo(name="main", ref="refs/heads/main", target_commit="main-sha")],
            converts=self.converts,
            tags=[],
        )

    def create_branch(  # type: ignore[override]
        self,
        *args: Any,
        branch: str = "",
        repo_type: str = "",
        revision: str = "",
        exist_ok: bool = False,
        **kwargs: Any,
    ) -> None:
        self.created_branches.append((branch, revision))


def test_create_branch_from_the_initial_commit_of_the_main_branch() -> None:
    calls: list[tuple[str, str, Any]] = []
    responses = {
        "HEAD": FakeResponse(headers={"X-Total-Count": "18"}),
        "GET": FakeResponse(headers={}, json_data=[{"id": "root-sha"}]),
    }
    committer_hf_api = RefsHfApi()
    with _patch_http_backoff(responses, calls):
        create_branch(
            dataset="user/dataset",
            target_revision="refs/convert/duckdb",
            hf_api=RefsHfApi(),
            committer_hf_api=committer_hf_api,
        )
    # the branch is created from the initial commit, and the requests are pinned to the sha of the main branch
    assert committer_hf_api.created_branches == [("refs/convert/duckdb", "root-sha")]
    assert [call[1] for call in calls] == ["https://huggingface.co/api/datasets/user/dataset/commits/main-sha"] * 2


def test_create_branch_does_nothing_if_the_branch_already_exists() -> None:
    calls: list[tuple[str, str, Any]] = []
    responses = {
        "HEAD": FakeResponse(headers={"X-Total-Count": "18"}),
        "GET": FakeResponse(headers={}, json_data=[{"id": "root-sha"}]),
    }
    parquet_ref = GitRefInfo(name="parquet", ref="refs/convert/parquet", target_commit="sha")
    committer_hf_api = RefsHfApi()
    with _patch_http_backoff(responses, calls):
        create_branch(
            dataset="user/dataset",
            target_revision="refs/convert/parquet",
            hf_api=RefsHfApi(converts=[parquet_ref]),
            committer_hf_api=committer_hf_api,
        )
    assert committer_hf_api.created_branches == []
    assert calls == []  # ^ no commit listing at all if the branch already exists
