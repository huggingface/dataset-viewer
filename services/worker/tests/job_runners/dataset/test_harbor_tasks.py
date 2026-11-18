# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import httpx
import pytest
from huggingface_hub import DatasetInfo, RepoFile, RepoFolder
from huggingface_hub.utils import RepositoryNotFoundError
from libcommon.dtos import Priority
from libcommon.exceptions import DatasetNotFoundError

from worker.config import AppConfig
from worker.job_runners.dataset.harbor_tasks import DatasetHarborTasksJobRunner, compute_harbor_tasks_response

DATASET = "org/harbor"
REVISION = "a" * 40
ENDPOINT = "https://huggingface.co"


@pytest.fixture
def hf_api() -> Iterator[MagicMock]:
    with patch("worker.job_runners.dataset.harbor_tasks.HfApi", autospec=True) as constructor:
        api = constructor.return_value
        api.dataset_info.return_value = DatasetInfo(id=DATASET, tags=["rl-environment", "harbor"])
        api.list_repo_tree.return_value = iter([])
        yield api


@pytest.mark.parametrize("library_tag", ["harbor", "library:harbor"])
def test_index_at_job_revision(hf_api: MagicMock, library_tag: str) -> None:
    hf_api.dataset_info.return_value = DatasetInfo(id=DATASET, tags=["rl-environment", library_tag])
    paths = [
        "tasks/b/task.toml",
        "tasks/b/environment/Dockerfile",
        "tasks/a/tests/test.sh",
        "tasks/a/README.md",
        "tasks/a/solution/solve.sh",
        "tasks/a/task.toml",
        "tasks/a/instruction.md",
        "tasks/a/environment/Dockerfile",
        "tasks/a/environment/data/input.txt",
        "not-a-task/instruction.md",
        "archive.tar.gz",
    ]
    hf_api.list_repo_tree.return_value = iter(
        [RepoFolder(path="tasks/fake/task.toml", oid="oid")]
        + [RepoFile(path=path, size=42, oid="oid") for path in paths]
    )
    runner = DatasetHarborTasksJobRunner(
        job_info={
            "type": "dataset-harbor-tasks",
            "params": {"dataset": DATASET, "revision": REVISION, "config": None, "split": None},
            "job_id": "job_id",
            "priority": Priority.NORMAL,
            "difficulty": 50,
            "started_at": None,
        },
        app_config=AppConfig(),
    )
    result = list(runner.compute())
    assert len(result) == 1
    assert result[0].progress == 1.0
    assert result[0].content == {
        "dataset": DATASET,
        "revision": REVISION,
        "tasks": [
            {
                "path": "tasks/a",
                "files": [
                    {"path": path, "size": 42}
                    for path in [
                        "tasks/a/instruction.md",
                        "tasks/a/environment/Dockerfile",
                        "tasks/a/environment/data/input.txt",
                        "tasks/a/solution/solve.sh",
                        "tasks/a/task.toml",
                        "tasks/a/tests/test.sh",
                    ]
                ],
            },
            {
                "path": "tasks/b",
                "files": [
                    {"path": "tasks/b/environment/Dockerfile", "size": 42},
                    {"path": "tasks/b/task.toml", "size": 42},
                ],
            },
        ],
    }
    hf_api.dataset_info.assert_called_once_with(DATASET, revision=REVISION, expand=["tags"])
    hf_api.list_repo_tree.assert_called_once_with(DATASET, repo_type="dataset", revision=REVISION, recursive=True)


def test_root_and_nested_tasks_own_their_files(hf_api: MagicMock) -> None:
    paths = [
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "environment/nested/task.toml",
        "environment/nested/instruction.md",
        "environment/nested/tests/test.sh",
        "environment/nested/README.md",
        "tasks/other/task.toml",
    ]
    hf_api.list_repo_tree.return_value = (RepoFile(path=path, size=1, oid="oid") for path in paths)
    result = compute_harbor_tasks_response(DATASET, REVISION, ENDPOINT)
    assert result["tasks"] == [
        {
            "path": "",
            "files": [{"path": path, "size": 1} for path in ["instruction.md", "environment/Dockerfile", "task.toml"]],
        },
        {
            "path": "environment/nested",
            "files": [
                {"path": path, "size": 1}
                for path in [
                    "environment/nested/instruction.md",
                    "environment/nested/task.toml",
                    "environment/nested/tests/test.sh",
                ]
            ],
        },
        {"path": "tasks/other", "files": [{"path": "tasks/other/task.toml", "size": 1}]},
    ]


def test_discovers_tasks_after_ten_thousand_entries(hf_api: MagicMock) -> None:
    paths = [f"data/{index}.txt" for index in range(10_001)] + ["tasks/last/task.toml"]
    hf_api.list_repo_tree.return_value = (RepoFile(path=path, size=1, oid="oid") for path in paths)
    assert compute_harbor_tasks_response(DATASET, REVISION, ENDPOINT)["tasks"] == [
        {"path": "tasks/last", "files": [{"path": "tasks/last/task.toml", "size": 1}]}
    ]


@pytest.mark.parametrize("tags", [None, [], ["harbor"], ["rl-environment"], ["rl-environment", "openenv"]])
def test_non_harbor_datasets_skip_tree(hf_api: MagicMock, tags: list[str] | None) -> None:
    hf_api.dataset_info.return_value = DatasetInfo(id=DATASET, tags=tags)
    assert compute_harbor_tasks_response(DATASET, REVISION, ENDPOINT)["tasks"] == []
    hf_api.list_repo_tree.assert_not_called()


def test_no_unpacked_tasks(hf_api: MagicMock) -> None:
    hf_api.list_repo_tree.return_value = iter([RepoFile(path="tasks.tar.gz", size=42, oid="oid")])
    assert compute_harbor_tasks_response(DATASET, REVISION, ENDPOINT)["tasks"] == []


def test_missing_dataset(hf_api: MagicMock) -> None:
    hf_api.dataset_info.side_effect = RepositoryNotFoundError(
        "missing", response=httpx.Response(404, request=httpx.Request("GET", ENDPOINT))
    )
    with pytest.raises(DatasetNotFoundError):
        compute_harbor_tasks_response(DATASET, REVISION, ENDPOINT)


def test_incomplete_tree_is_not_cached_as_success(hf_api: MagicMock) -> None:
    def failing_tree() -> Iterator[RepoFile]:
        yield RepoFile(path="tasks/first/task.toml", size=42, oid="oid")
        raise ConnectionError("next tree page failed")

    hf_api.list_repo_tree.return_value = failing_tree()
    with pytest.raises(ConnectionError, match="next tree page failed"):
        compute_harbor_tasks_response(DATASET, REVISION, ENDPOINT)
