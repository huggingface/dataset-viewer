# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from time import sleep
from typing import Literal, Optional, TypedDict
from urllib import parse

from huggingface_hub import HfApi
from huggingface_hub.constants import REPO_TYPE_DATASET
from libcommon.simple_cache import get_datasets_with_last_updated_kind

PARQUET_CACHE_KIND = "config-parquet"
DAYS = 1


class ParquetCounters(TypedDict):
    datasets: int
    new_discussions: int
    dismissed_discussions: int
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
    logging.info("Create a Hub discussion to notify about parquet conversion")
    datasets = limit_to_one_dataset_per_namespace(
        get_datasets_with_last_updated_kind(kind=PARQUET_CACHE_KIND, days=DAYS)
    )

    logging.info(f"Creating discussions for {len(datasets)} datasets")
    log_batch = 100
    counters: ParquetCounters = {
        "datasets": 0,
        "new_discussions": 0,
        "dismissed_discussions": 0,
        "errors": 0,
    }

    def get_log() -> str:
        return (
            f" {counters['new_discussions']} discussions have been opened. A total of"
            f" {len(datasets)} datasets were selected, but {counters['dismissed_discussions']} datasets"
            f" already had a discussion (open or closed). {counters['errors']} errors."
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
                counters["dismissed_discussions"] += 1
                continue
            else:
                hf_api.create_discussion(
                    repo_id=dataset,
                    repo_type=REPO_TYPE_DATASET,
                    title="[bot] Conversion to Parquet",
                    description=create_discussion_description(
                        dataset=dataset,
                        hf_endpoint=hf_endpoint,
                        parquet_revision=parquet_revision,
                        bot_associated_user_name=bot_associated_user_name,
                    ),
                    token=bot_token,
                )
                sleep(1)
                # ^ see https://github.com/huggingface/moon-landing/issues/7729 (internal)
                counters["new_discussions"] += 1

        except Exception as e:
            logging.warning(f"Failed to post a message for {dataset}: {e}")
            counters["errors"] += 1

        logging.debug(get_log())
        if (counters["datasets"]) % log_batch == 0:
            logging.info(get_log())

    logging.info(get_log())
    logging.info("All the messages about parquet conversion have been posted.")

    return counters


def create_discussion_description(
    dataset: str, hf_endpoint: str, parquet_revision: str, bot_associated_user_name: str
) -> str:
    link_parquet = create_link(
        text=parquet_revision,
        dataset=dataset,
        hf_endpoint=hf_endpoint,
        revision_type="tree",
        revision=parquet_revision,
    )
    return (
        f"The {bot_associated_user_name} bot has created a version of this dataset in the [Parquet"
        " format](https://parquet.apache.org/). You can learn more about the advantages associated with this format"
        f""" in the [documentation](https://huggingface.co/docs/datasets-server/parquet).

The Parquet files are published in the {link_parquet} branch."""
    )


def create_link(
    text: str, dataset: str, hf_endpoint: str, revision_type: Literal["commit", "tree"], revision: str
) -> str:
    return f"[`{text}`]({hf_endpoint}/datasets/{dataset}/{revision_type}/{parse.quote(revision, safe='')})"


def limit_to_one_dataset_per_namespace(datasets: list[str]) -> list[str]:
    """
    Limit the number of datasets to one per namespace.

    For instance, if we have `a/b` and `a/c`, we will only keep one of them.
    The choice is arbitrary. The filtered list has no particular order.

    Args:
        datasets (list[str]): The list of datasets to filter.

    Returns:
        list[str]: The filtered list of datasets.
    """
    namespaces: set[str] = set()
    selected_datasets: list[str] = []
    for dataset in datasets:
        namespace = get_namespace(dataset)
        if (namespace is None) or (namespace in namespaces):
            continue
        namespaces.add(namespace)
        selected_datasets.append(dataset)
    return selected_datasets


def get_namespace(dataset: str) -> Optional[str]:
    splits = dataset.split("/")
    return splits[0] if len(splits) == 2 else None
