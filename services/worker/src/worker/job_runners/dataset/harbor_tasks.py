# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import posixpath
from collections.abc import Iterator
from typing import Optional

from huggingface_hub import HfApi, RepoFile
from huggingface_hub.utils import RepositoryNotFoundError
from libcommon.exceptions import DatasetNotFoundError

from worker.dtos import CompleteJobResult, DatasetHarborTasksResponse, HarborTask, HarborTaskFile
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner


def compute_harbor_tasks_response(
    dataset: str,
    revision: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> DatasetHarborTasksResponse:
    api = HfApi(endpoint=hf_endpoint, token=hf_token)
    response = DatasetHarborTasksResponse(dataset=dataset, revision=revision, tasks=[])
    try:
        tags = api.dataset_info(dataset, revision=revision, expand=["tags"]).tags or []
        if "rl-environment" not in tags or not {"harbor", "library:harbor"}.intersection(tags):
            return response
        files = [
            entry
            for entry in api.list_repo_tree(dataset, repo_type="dataset", revision=revision, recursive=True)
            if isinstance(entry, RepoFile)
        ]
    except RepositoryNotFoundError as err:
        raise DatasetNotFoundError(f"Cannot get Harbor tasks for {dataset=}") from err

    tasks: dict[str, HarborTask] = {
        posixpath.dirname(file.path): HarborTask(path=posixpath.dirname(file.path), files=[])
        for file in files
        if posixpath.basename(file.path) == "task.toml"
    }
    for file in sorted(files, key=lambda file: file.path):
        task_path = posixpath.dirname(file.path)
        while task_path and task_path not in tasks:
            task_path = posixpath.dirname(task_path)
        if task_path not in tasks:
            continue
        relative_path = posixpath.relpath(file.path, task_path or ".")
        if relative_path in {"task.toml", "instruction.md"} or relative_path.startswith(
            ("environment/", "solution/", "tests/")
        ):
            task_files = tasks[task_path]["files"]
            task_files.insert(
                0 if relative_path == "instruction.md" else len(task_files),
                HarborTaskFile(path=file.path, size=file.size),
            )
    response["tasks"] = [tasks[path] for path in sorted(tasks)]
    return response


class DatasetHarborTasksJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-harbor-tasks"

    def compute(self) -> Iterator[CompleteJobResult]:
        yield CompleteJobResult(
            compute_harbor_tasks_response(
                dataset=self.dataset,
                revision=self.dataset_git_revision,
                hf_endpoint=self.app_config.common.hf_endpoint,
                hf_token=self.app_config.common.hf_token,
            )
        )
