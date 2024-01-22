# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from dataclasses import dataclass
from typing import Literal, Optional
from urllib import parse

from huggingface_hub import HfApi
from huggingface_hub.constants import REPO_TYPE_DATASET
from libcommon.simple_cache import get_datasets_with_last_updated_kind

PARQUET_CACHE_KIND = "config-parquet"
DAYS = 1


@dataclass
class ParquetCounters:
    datasets: int = 0
    new_discussions: int = 0
    dismissed_discussions: int = 0
    errors: int = 0


@dataclass
class Counters:
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
    counters = ParquetCounters()

    def get_log() -> str:
        return (
            f" [{counters.datasets}/{len(datasets)}] {counters.new_discussions} discussions"
            f" have been opened, {counters.dismissed_discussions} datasets"
            f" already had a discussion (open or closed). {counters.errors} errors."
        )

    hf_api = HfApi(endpoint=hf_endpoint, token=bot_token)

    for dataset in datasets:
        counters.datasets += 1
        prefix = f"[{counters.datasets}/{len(datasets)}]"
        logging.info(f"{prefix} Processing dataset {dataset}")
        try:
            try:
                next(
                    hf_api.get_repo_discussions(
                        repo_id=dataset, repo_type=REPO_TYPE_DATASET, token=bot_token, author=bot_associated_user_name
                    )
                )
                # if we get here, the bot has already opened a discussion for this dataset
                counters.dismissed_discussions += 1
                logging.info(f"{prefix} [dismissed] Dataset {dataset} already has a discussion, skipping")
            except StopIteration:
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
                counters.new_discussions += 1
                logging.info(f"{prefix} [new] Dataset {dataset} has a new discussion")
        except Exception as e:
            counters.errors += 1
            logging.warning(f"{prefix} [error] Failed to process dataset {dataset}: {e}")

        logging.debug(get_log())
        if (counters.datasets) % log_batch == 0:
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
