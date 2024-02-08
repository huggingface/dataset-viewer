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

DISCUSSION_TITLE = "[bot] [No action needed] Conversion to Parquet"
DISCUSSION_DESCRIPTION = """The {bot_name} bot has created a version of this dataset in the Parquet format in the {parquet_link} branch.

## What is Parquet?

Parquet is a columnar storage format optimized for querying and processing large datasets. Parquet is a popular choice for big data processing and analytics and is widely used for data processing and machine learning.

In Parquet, data is divided in chunks called "row groups", and within each row group, it is stored in columns rather than rows. Each row group column is compressed separately using the most adapted compression algorithm, and contains metadata about the data it contains.

This structure allows for efficient reading and querying of the data:
- only the necessary columns are read from disk (projection pushdown); no need to read the entire file. This reduces the memory requirement for working with Parquet data. 
- thanks to the statistics of each row group (min, max, number of NULL values), entire row groups can be skipped if they do not contain the data of interest (automatic filtering)
- the data is compressed, which reduces the amount of data that needs to be stored and transfered

Note that a Parquet file contains a single table. If a dataset has multiple tables (e.g. multiple splits or configurations), each table is stored in a separate Parquet file.

You can learn more about the advantages associated with this format in the [documentation](https://huggingface.co/docs/datasets-server/parquet).

## How to access the Parquet version of the dataset?

You can access the Parquet version of the dataset by following this link: {parquet_link}

## What should I do?

You don't need to do anything. The Parquet version of the dataset is available for you to use. Refer to the [documentation](https://huggingface.co/docs/datasets-server/parquet_process) for examples and code snippets on how to query the Parquet files with ClickHouse, DuckDB, Pandas or Polars.

If you have any questions or concerns, feel free to ask in the discussion below. You can also close the discussion if you don't have any questions."""


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
    parquet_link = create_link(
        text=parquet_revision,
        dataset=dataset,
        hf_endpoint=hf_endpoint,
        revision_type="tree",
        revision=parquet_revision,
    )
    return DISCUSSION_DESCRIPTION.format(bot_name=bot_associated_user_name, parquet_link=parquet_link)


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
        datasets (`list[str]`): The list of datasets to filter.

    Returns:
        `list[str]`: The filtered list of datasets.
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
