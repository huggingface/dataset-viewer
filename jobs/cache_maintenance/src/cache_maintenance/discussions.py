# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from typing import Optional, TypedDict
from urllib import parse

from huggingface_hub import HfApi
from libcommon.simple_cache import get_datasets_with_last_updated_kind
from huggingface_hub.constants import REPO_TYPE_DATASET

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
    datasets = limit_to_one_dataset_per_namespace(
        get_datasets_with_last_updated_kind(kind=PARQUET_CACHE_KIND, days=DAYS)
    )

    logging.info(f"Posting messages for {len(datasets)} datasets")
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
            f" {len(datasets)} datasets): {counters['new_discussions']} discussions have been opened."
            f" {counters['dismissed_messages']} messages have been dismissed because the discussion had been closed."
            f" {counters['errors']} errors."
        )

    hf_api = HfApi(endpoint=hf_endpoint, token=bot_token)

    for dataset in datasets:
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
                if discussion.status == CLOSED_STATUS:
                    counters["dismissed_messages"] += 1
                    continue
                hf_api.comment_discussion(
                    repo_id=dataset,
                    repo_type=REPO_TYPE_DATASET,
                    discussion_num=discussion.num,
                    comment=create_following_comment(
                        dataset=dataset, hf_endpoint=hf_endpoint, parquet_revision=parquet_revision
                    ),
                    token=bot_token,
                )
            else:
                discussion = hf_api.create_discussion(
                    repo_id=dataset,
                    repo_type=REPO_TYPE_DATASET,
                    title="Notifications from Datasets Server",
                    description=create_first_comment(
                        dataset=dataset, hf_endpoint=hf_endpoint, parquet_revision=parquet_revision
                    ),
                    token=bot_token,
                )
                counters["new_discussions"] += 1

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


def create_first_comment(dataset: str, hf_endpoint: str, parquet_revision: str) -> str:
    following_comment = create_following_comment(
        dataset=dataset, hf_endpoint=hf_endpoint, parquet_revision=parquet_revision
    )
    return (
        "Datasets can be published in any format (CSV, JSONL, directories of images, etc.) to the Hub, and they "
        "are easily accessed with the 🤗 Datasets library. For a more performant experience (especially when it "
        f"""comes to large datasets), {following_comment}

_The Datasets Server bot will provide updates in the comments. Close the discussion if you want to stop receiving"""
        " notifications._"
    )


def create_following_comment(dataset: str, hf_endpoint: str, parquet_revision: str) -> str:
    link = create_link(dataset=dataset, hf_endpoint=hf_endpoint, parquet_revision=parquet_revision)
    return (
        f"""Datasets Server has automatically converted this dataset to the [Parquet format](https://parquet.apache.org/).

The Parquet files are published to the Hub in the {link} branch.

You can find more details about the Parquet conversion in the"""
        " [documentation](https://huggingface.co/docs/datasets-server/parquet)."
    )


def create_link(dataset: str, hf_endpoint: str, parquet_revision: str) -> str:
    return f"[`{parquet_revision}`]({hf_endpoint}/datasets/{dataset}/tree/{parse.quote(parquet_revision, safe='')})"


def limit_to_one_dataset_per_namespace(datasets: set[str]) -> set[str]:
    """
    Limit the number of datasets to one per namespace.

    For instance, if we have `a/b` and `a/c`, we will only keep one of them.
    The choice is arbitrary.

    Args:
        datasets: The set of datasets to filter.

    Returns:
        The filtered set of datasets.
    """
    namespaces: set[str] = set()
    selected_datasets: set[str] = set()
    for dataset in datasets:
        namespace = get_namespace(dataset)
        if (namespace is None) or (namespace in namespaces):
            continue
        namespaces.add(namespace)
        selected_datasets.add(dataset)
    return selected_datasets


def get_namespace(dataset: str) -> Optional[str]:
    splits = dataset.split("/")
    return splits[0] if len(splits) == 2 else None
