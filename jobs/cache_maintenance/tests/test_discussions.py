# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from http import HTTPStatus

import pytest
from huggingface_hub.community import DiscussionComment
from libcommon.simple_cache import DatasetWithRevision, upsert_response
from libcommon.utils import get_datetime

from cache_maintenance.config import JobConfig
from cache_maintenance.discussions import (
    DAYS,
    PARQUET_CACHE_KIND,
    create_discussion_description,
    create_link,
    create_parquet_comment,
    limit_to_one_dataset_per_namespace,
    post_messages,
)

from .utils import (
    REVISION_NAME,
    TemporaryDataset,
    close_discussion,
    count_comments,
    fetch_bot_discussion,
)


@pytest.mark.parametrize(
    "datasets, valid_expected_datasets",
    [
        (set(), [set()]),
        ({"a/b"}, [{"a/b"}]),
        ({"a"}, [set()]),
        ({"a/b/c"}, [set()]),
        ({"a/b", "a/b"}, [{"a/b"}]),
        ({"a/b", "a/c"}, [{"a/b"}, {"a/c"}]),
        ({"a/b", "b/b"}, [{"a/b", "b/b"}]),
        ({"a/b", "b"}, [{"a/b"}]),
    ],
)
def test_limit_to_one_dataset_per_namespace(datasets: set[str], valid_expected_datasets: list[set[str]]) -> None:
    assert any(
        {
            d.dataset
            for d in limit_to_one_dataset_per_namespace(
                datasets_with_revision=[
                    DatasetWithRevision(dataset=dataset, revision=REVISION_NAME) for dataset in datasets
                ]
            )
        }
        == expected_datasets
        for expected_datasets in valid_expected_datasets
    )


def test_create_link() -> None:
    assert (
        create_link(
            text="sometext",
            dataset="a/b",
            hf_endpoint="https://huggingface.co",
            revision_type="commit",
            revision="c/d",
        )
        == "[`sometext`](https://huggingface.co/datasets/a/b/commit/c%2Fd)"
    )


def test_post_messages_in_one_dataset(job_config: JobConfig) -> None:
    with TemporaryDataset(prefix="dataset") as dataset:
        assert fetch_bot_discussion(dataset=dataset.repo_id) is None
        # set "config-parquet" entry for the dataset
        first_revision = "3bb24dcad2b45b45e20fc0accc93058dcbe8087d"
        upsert_response(
            kind=PARQUET_CACHE_KIND,
            dataset=dataset.repo_id,
            content={},
            http_status=HTTPStatus.OK,
            dataset_git_revision=first_revision,
        )
        # call post_messages
        counters = post_messages(
            hf_endpoint=job_config.common.hf_endpoint,
            bot_associated_user_name=job_config.discussions.bot_associated_user_name,
            bot_token=job_config.discussions.bot_token,
            parquet_revision=job_config.discussions.parquet_revision,
        )
        # ensure one message has been posted in a dataset discussion
        assert counters["parquet"] == {
            "datasets": 1,
            "messages": 1,
            "new_discussions": 1,
            "errors": 0,
            "dismissed_messages": 0,
        }
        first_discussion = fetch_bot_discussion(dataset=dataset.repo_id)
        assert first_discussion is not None
        assert count_comments(first_discussion) == 2
        first_comment = first_discussion.events[0]
        assert isinstance(first_comment, DiscussionComment)
        assert first_comment.content == create_discussion_description()
        second_comment = first_discussion.events[1]
        assert isinstance(second_comment, DiscussionComment)
        assert second_comment.content == create_parquet_comment(
            dataset=dataset.repo_id,
            hf_endpoint=job_config.common.hf_endpoint,
            parquet_revision=job_config.discussions.parquet_revision,
            dataset_revision=first_revision,
        )
        # set a new "config-parquet" entry for the dataset
        second_revision = "9a0bd9fe2a87bbb82702ed170a53cf4e86535070"
        upsert_response(
            kind=PARQUET_CACHE_KIND,
            dataset=dataset.repo_id,
            content={},
            http_status=HTTPStatus.OK,
            dataset_git_revision=second_revision,
        )
        # call post_messages again
        counters = post_messages(
            hf_endpoint=job_config.common.hf_endpoint,
            bot_associated_user_name=job_config.discussions.bot_associated_user_name,
            bot_token=job_config.discussions.bot_token,
            parquet_revision=job_config.discussions.parquet_revision,
        )
        # ensure the message have been posted in the same discussion
        assert counters["parquet"] == {
            "datasets": 1,
            "messages": 1,
            "new_discussions": 0,
            "errors": 0,
            "dismissed_messages": 0,
        }
        second_discussion = fetch_bot_discussion(dataset=dataset.repo_id)
        assert second_discussion is not None
        assert first_discussion.num == second_discussion.num
        assert count_comments(second_discussion) == 3
        third_comment = second_discussion.events[2]
        assert isinstance(third_comment, DiscussionComment)
        assert third_comment.content == create_parquet_comment(
            dataset=dataset.repo_id,
            hf_endpoint=job_config.common.hf_endpoint,
            parquet_revision=job_config.discussions.parquet_revision,
            dataset_revision=second_revision,
        )
        # close the discussion
        close_discussion(dataset=dataset.repo_id, discussion_num=first_discussion.num)
        # call post_messages again
        counters = post_messages(
            hf_endpoint=job_config.common.hf_endpoint,
            bot_associated_user_name=job_config.discussions.bot_associated_user_name,
            bot_token=job_config.discussions.bot_token,
            parquet_revision=job_config.discussions.parquet_revision,
        )
        # ensure the message has not been posted
        assert counters["parquet"] == {
            "datasets": 1,
            "messages": 0,
            "new_discussions": 0,
            "errors": 0,
            "dismissed_messages": 1,
        }
        third_discussion = fetch_bot_discussion(dataset=dataset.repo_id)
        assert third_discussion is not None
        assert first_discussion.num == third_discussion.num
        assert count_comments(third_discussion) == 3


def test_post_messages_with_two_datasets_in_one_namespace(job_config: JobConfig) -> None:
    with TemporaryDataset(prefix="dataset1") as dataset1, TemporaryDataset(prefix="dataset2") as dataset2:
        assert fetch_bot_discussion(dataset=dataset1.repo_id) is None
        assert fetch_bot_discussion(dataset=dataset2.repo_id) is None
        # set "config-parquet" entry for the two datasets
        upsert_response(
            kind=PARQUET_CACHE_KIND,
            dataset=dataset1.repo_id,
            dataset_git_revision=REVISION_NAME,
            content={},
            http_status=HTTPStatus.OK,
        )
        upsert_response(
            kind=PARQUET_CACHE_KIND,
            dataset=dataset2.repo_id,
            dataset_git_revision=REVISION_NAME,
            content={},
            http_status=HTTPStatus.OK,
        )
        # call post_messages
        counters = post_messages(
            hf_endpoint=job_config.common.hf_endpoint,
            bot_associated_user_name=job_config.discussions.bot_associated_user_name,
            bot_token=job_config.discussions.bot_token,
            parquet_revision=job_config.discussions.parquet_revision,
        )
        # ensure one message has been posted in only one dataset discussion
        assert counters["parquet"] == {
            "datasets": 1,
            "messages": 1,
            "new_discussions": 1,
            "errors": 0,
            "dismissed_messages": 0,
        }
        discussion1 = fetch_bot_discussion(dataset=dataset1.repo_id)
        discussion2 = fetch_bot_discussion(dataset=dataset2.repo_id)
        discussion = discussion1 or discussion2
        assert discussion is not None
        assert discussion1 is None or discussion2 is None
        assert count_comments(discussion) == 2
        comment = discussion.events[1]
        assert isinstance(comment, DiscussionComment)
        assert comment.content == create_parquet_comment(
            dataset=dataset1.repo_id,
            hf_endpoint=job_config.common.hf_endpoint,
            parquet_revision=job_config.discussions.parquet_revision,
            dataset_revision=None,
        ) or create_parquet_comment(
            dataset=dataset2.repo_id,
            hf_endpoint=job_config.common.hf_endpoint,
            parquet_revision=job_config.discussions.parquet_revision,
            dataset_revision=None,
        )


@pytest.mark.parametrize(
    "gated,private",
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_post_messages_in_private_or_gated_dataset(job_config: JobConfig, gated: bool, private: bool) -> None:
    with TemporaryDataset(prefix="dataset", gated=gated, private=private) as dataset:
        assert fetch_bot_discussion(dataset=dataset.repo_id) is None
        # set "config-parquet" entry for the dataset
        upsert_response(
            kind=PARQUET_CACHE_KIND,
            dataset=dataset.repo_id,
            dataset_git_revision=REVISION_NAME,
            content={},
            http_status=HTTPStatus.OK,
        )
        # call post_messages
        counters = post_messages(
            hf_endpoint=job_config.common.hf_endpoint,
            bot_associated_user_name=job_config.discussions.bot_associated_user_name,
            bot_token=job_config.discussions.bot_token,
            parquet_revision=job_config.discussions.parquet_revision,
        )
        # ensure one message has been posted in a dataset discussion
        # YES: even if it's private. Should we forbid this?
        # Normally: the cache should not contain private datasets, but a public
        # dataset can be switched to private, and for some reason, or during some
        # time, the cache can contain private datasets.
        assert counters["parquet"] == {
            "datasets": 1,
            "messages": 1,
            "new_discussions": 1,
            "errors": 0,
            "dismissed_messages": 0,
        }
        first_discussion = fetch_bot_discussion(dataset=dataset.repo_id)
        assert first_discussion is not None
        assert count_comments(first_discussion) == 2
        comment = first_discussion.events[1]
        assert isinstance(comment, DiscussionComment)
        assert comment.content == create_parquet_comment(
            dataset=dataset.repo_id,
            hf_endpoint=job_config.common.hf_endpoint,
            parquet_revision=job_config.discussions.parquet_revision,
            dataset_revision=REVISION_NAME,
        )


def test_post_messages_for_outdated_response(job_config: JobConfig) -> None:
    with TemporaryDataset(prefix="dataset") as dataset:
        assert fetch_bot_discussion(dataset=dataset.repo_id) is None
        # set "config-parquet" entry for the dataset
        upsert_response(
            kind=PARQUET_CACHE_KIND,
            dataset=dataset.repo_id,
            dataset_git_revision=REVISION_NAME,
            content={},
            http_status=HTTPStatus.OK,
            updated_at=get_datetime(days=DAYS + 10),
        )
        # call post_messages
        counters = post_messages(
            hf_endpoint=job_config.common.hf_endpoint,
            bot_associated_user_name=job_config.discussions.bot_associated_user_name,
            bot_token=job_config.discussions.bot_token,
            parquet_revision=job_config.discussions.parquet_revision,
        )
        # ensure no discussion has been created, because the content was too old
        assert counters["parquet"] == {
            "datasets": 0,
            "messages": 0,
            "new_discussions": 0,
            "errors": 0,
            "dismissed_messages": 0,
        }
        assert fetch_bot_discussion(dataset=dataset.repo_id) is None
        # update the content
        upsert_response(
            kind=PARQUET_CACHE_KIND,
            dataset=dataset.repo_id,
            dataset_git_revision=REVISION_NAME,
            content={},
            http_status=HTTPStatus.OK,
        )
        # call post_messages
        counters = post_messages(
            hf_endpoint=job_config.common.hf_endpoint,
            bot_associated_user_name=job_config.discussions.bot_associated_user_name,
            bot_token=job_config.discussions.bot_token,
            parquet_revision=job_config.discussions.parquet_revision,
        )
        # ensure one message has been posted in a dataset discussion
        assert counters["parquet"] == {
            "datasets": 1,
            "messages": 1,
            "new_discussions": 1,
            "errors": 0,
            "dismissed_messages": 0,
        }
        first_discussion = fetch_bot_discussion(dataset=dataset.repo_id)
        assert first_discussion is not None
        assert count_comments(first_discussion) == 2
        comment = first_discussion.events[1]
        assert isinstance(comment, DiscussionComment)
        assert comment.content == create_parquet_comment(
            dataset=dataset.repo_id,
            hf_endpoint=job_config.common.hf_endpoint,
            parquet_revision=job_config.discussions.parquet_revision,
            dataset_revision=REVISION_NAME,
        )
