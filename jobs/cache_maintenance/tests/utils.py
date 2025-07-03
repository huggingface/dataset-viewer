# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import requests
from huggingface_hub import HfApi
from huggingface_hub.community import DiscussionComment, DiscussionWithDetails
from huggingface_hub.constants import (
    REPO_TYPE_DATASET,
    REPO_TYPES,
    REPO_TYPES_URL_PREFIXES,
)
from huggingface_hub.hf_api import HfApi
from libcommon.resources import Resource

from .constants import (
    CI_HUB_ENDPOINT,
    CI_PARQUET_CONVERTER_USER,
    CI_USER,
    CI_USER_TOKEN,
)

DATASET = "dataset"

REVISION_NAME = "revision"

hf_api = HfApi(endpoint=CI_HUB_ENDPOINT)


def get_default_config_split() -> tuple[str, str]:
    config = "default"
    split = "train"
    return config, split

def create_empty_hub_dataset_repo(
    *,
    prefix: str,
    file_paths: Optional[list[str]] = None,
    private: bool = False,
    gated: Optional[str] = None,
) -> str:
    dataset_name = f"{prefix}-{int(time.time() * 10e3)}"
    repo_id = f"{CI_USER}/{dataset_name}"
    hf_api.create_repo(repo_id=repo_id, token=CI_USER_TOKEN, repo_type=DATASET, private=private)
    if gated:
        HfApi(endpoint=CI_HUB_ENDPOINT).update_repo_settings(
            repo_id=repo_id,
            token=CI_USER_TOKEN,
            gated=gated,
            repo_type=DATASET,
        )
    if file_paths is not None:
        for file_path in file_paths:
            hf_api.upload_file(
                token=CI_USER_TOKEN,
                path_or_fileobj=file_path,
                path_in_repo=Path(file_path).name.replace("{dataset_name}", dataset_name),
                repo_id=repo_id,
                repo_type=DATASET,
            )
    return repo_id


def delete_hub_dataset_repo(repo_id: str) -> None:
    with suppress(requests.exceptions.HTTPError, ValueError):
        hf_api.delete_repo(repo_id=repo_id, token=CI_USER_TOKEN, repo_type=DATASET)


@dataclass
class TemporaryDataset(Resource):
    """A temporary dataset."""

    prefix: str
    gated: bool = False
    private: bool = False

    repo_id: str = field(init=False)

    def allocate(self) -> None:
        self.repo_id = create_empty_hub_dataset_repo(
            prefix=self.prefix, gated="auto" if self.gated else None, private=self.private
        )

    def release(self) -> None:
        delete_hub_dataset_repo(repo_id=self.repo_id)


def fetch_bot_discussion(dataset: str) -> Optional[DiscussionWithDetails]:
    """
    Fetch the discussion for a dataset and a user.
    """
    hf_api = HfApi(endpoint=CI_HUB_ENDPOINT, token=CI_USER_TOKEN)
    discussions = hf_api.get_repo_discussions(repo_id=dataset, repo_type=REPO_TYPE_DATASET)
    discussion = next(
        (discussion for discussion in discussions if discussion.author == CI_PARQUET_CONVERTER_USER), None
    )
    if not discussion:
        return None
    return hf_api.get_discussion_details(repo_id=dataset, repo_type=REPO_TYPE_DATASET, discussion_num=discussion.num)


def close_discussion(dataset: str, discussion_num: int) -> None:
    """
    Let the dataset owner close a discussion.
    """
    hf_api = HfApi(endpoint=CI_HUB_ENDPOINT, token=CI_USER_TOKEN)
    hf_api.change_discussion_status(
        repo_id=dataset, repo_type=REPO_TYPE_DATASET, discussion_num=discussion_num, new_status="closed"
    )


def count_comments(discussion: DiscussionWithDetails) -> int:
    return len(list(event for event in discussion.events if isinstance(event, DiscussionComment)))
