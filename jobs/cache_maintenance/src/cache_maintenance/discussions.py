# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from typing import Literal, Optional, TypedDict
from urllib import parse

from huggingface_hub import HfApi
from huggingface_hub.constants import REPO_TYPE_DATASET
from libcommon.simple_cache import (
    DatasetWithRevision,
    get_datasets_with_last_updated_kind,
)

PARQUET_CACHE_KIND = "config-parquet"
DAYS = 1
CLOSED_STATUS = "closed"


class ParquetCounters(TypedDict):
    datasets: int
    messages: int
    dismissed_messages: int
    new_discussions: int
    errors: int


class Counters(TypedDict):
    parquet: ParquetCounters


def post_messages(
    hf_endpoint: str, bot_associated_user_name: Optional[str], bot_token: Optional[str], parquet_revision: str
) -> Counters:
    """
    Post messages in Hub discussions to notify users.
    """
    if not (bot_associated_user_name and bot_token):
        raise Exception("No bot token or user name provided, skipping posting messages.")

    return Counters(
        parquet=post_messages_on_parquet_conversion(
            hf_endpoint=hf_endpoint,
            bot_associated_user_name=bot_associated_user_name,
            bot_token=bot_token,
            parquet_revision=parquet_revision,
        )
    )


def post_messages_on_parquet_conversion(
    hf_endpoint: str,
    bot_associated_user_name: str,
    bot_token: str,
    parquet_revision: str,
) -> ParquetCounters:
    logging.info("Post messages in Hub discussion to notify about parquet conversion")
    datasets_with_revision = limit_to_one_dataset_per_namespace(
        get_datasets_with_last_updated_kind(kind=PARQUET_CACHE_KIND, days=DAYS)
    )

    logging.info(f"Posting messages for {len(datasets_with_revision)} datasets")
    log_batch = 100
    counters: ParquetCounters = {
        "datasets": 0,
        "messages": 0,
        "dismissed_messages": 0,
        "new_discussions": 0,
        "errors": 0,
    }

    def get_log() -> str:
        return (
            f"{counters['messages'] } messages posted (total:"
            f" {len(datasets_with_revision)} datasets): {counters['new_discussions']} discussions have been opened."
            f" {counters['dismissed_messages']} messages have been dismissed because the discussion had been closed."
            f" {counters['errors']} errors."
        )

    hf_api = HfApi(endpoint=hf_endpoint, token=bot_token)

    for dataset_with_revision in datasets_with_revision:
        dataset = dataset_with_revision.dataset
        revision = dataset_with_revision.revision

        counters["datasets"] += 1
        try:
            bot_discussions = [
                discussion
                for discussion in hf_api.get_repo_discussions(
                    repo_id=dataset, repo_type=REPO_TYPE_DATASET, token=bot_token
                )
                if discussion.author == bot_associated_user_name
            ]

            if bot_discussions:
                if len(bot_discussions) > 1:
                    logging.warning(
                        f"Found {len(bot_discussions)} discussions for {dataset} with bot {bot_associated_user_name},"
                        " only the first one will be used."
                    )
                discussion = bot_discussions[0]
            else:
                discussion = hf_api.create_discussion(
                    repo_id=dataset,
                    repo_type=REPO_TYPE_DATASET,
                    title="Notifications from Datasets Server",
                    description=create_discussion_description(),
                    token=bot_token,
                )
                counters["new_discussions"] += 1
            if discussion.status == CLOSED_STATUS:
                counters["dismissed_messages"] += 1
                continue
            hf_api.comment_discussion(
                repo_id=dataset,
                repo_type=REPO_TYPE_DATASET,
                discussion_num=discussion.num,
                comment=create_parquet_comment(
                    dataset=dataset,
                    hf_endpoint=hf_endpoint,
                    parquet_revision=parquet_revision,
                    dataset_revision=revision,
                ),
                token=bot_token,
            )

            counters["messages"] += 1

        except Exception as e:
            logging.warning(f"Failed to post a message for {dataset}: {e}")
            counters["errors"] += 1

        logging.debug(get_log())
        if (counters["datasets"]) % log_batch == 0:
            logging.info(get_log())

    logging.info(get_log())
    logging.info("All the messages about parquet conversion have been posted.")

    return counters


def temporary_call_to_action_for_feedback() -> str:
    return "Please comment below if you have any questions or feedback about this new notifications channel. "


def create_discussion_description() -> str:
    return (
        "The Datasets Server bot will post messages here about operations such as conversion to"
        " Parquet. There are some advantages associated with having a version of your dataset available in the "
        "[Parquet format](https://parquet.apache.org/). You can learn more about these in the"
        f""" [documentation](https://huggingface.co/docs/datasets-server/parquet).

_{temporary_call_to_action_for_feedback()}Close the discussion if you want to stop receiving notifications._"""
    )


def create_parquet_comment(
    dataset: str, hf_endpoint: str, parquet_revision: str, dataset_revision: Optional[str]
) -> str:
    link_dataset = f" revision {dataset_revision[:7]}" if dataset_revision else ""

    link_parquet = create_link(
        text=parquet_revision,
        dataset=dataset,
        hf_endpoint=hf_endpoint,
        revision_type="tree",
        revision=parquet_revision,
    )
    return f"""Datasets Server has converted the dataset{link_dataset} to Parquet.

The Parquet files are published to the Hub in the {link_parquet} branch."""


def create_link(
    text: str, dataset: str, hf_endpoint: str, revision_type: Literal["commit", "tree"], revision: str
) -> str:
    return f"[`{text}`]({hf_endpoint}/datasets/{dataset}/{revision_type}/{parse.quote(revision, safe='')})"


def limit_to_one_dataset_per_namespace(datasets_with_revision: list[DatasetWithRevision]) -> list[DatasetWithRevision]:
    """
    Limit the number of datasets to one per namespace.

    For instance, if we have `a/b` and `a/c`, we will only keep one of them.
    The choice is arbitrary. The filtered list has no particular order.

    Args:
        datasets (list[DatasetWithRevision]): The list of datasets (with revision) to filter.

    Returns:
        list[DatasetWithRevision]: The filtered list of datasets (with revision).
    """
    namespaces: set[str] = set()
    selected_datasets_with_revision: list[DatasetWithRevision] = []
    for dataset_with_revision in datasets_with_revision:
        namespace = get_namespace(dataset_with_revision.dataset)
        if (namespace is None) or (namespace in namespaces):
            continue
        namespaces.add(namespace)
        selected_datasets_with_revision.append(dataset_with_revision)
    return selected_datasets_with_revision


def get_namespace(dataset: str) -> Optional[str]:
    splits = dataset.split("/")
    return splits[0] if len(splits) == 2 else None
