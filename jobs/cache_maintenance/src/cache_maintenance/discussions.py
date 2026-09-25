# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Literal, Optional
from urllib import parse

from huggingface_hub import HfApi
from huggingface_hub.constants import REPO_TYPE_DATASET
from libcommon.constants import DATASET_INFO_KIND
from libcommon.simple_cache import (
    CachedArtifactNotFoundError,
    get_datasets_with_last_updated_kind,
    get_response,
)

PARQUET_CACHE_KIND = "config-parquet"
# the name of the datasets-builder used when the data is already stored as Parquet files on the Hub
PARQUET_BUILDER_NAME = "parquet"
DAYS = 1

DISCUSSION_TITLE = "[bot] [No action needed] Conversion to Parquet"
DISCUSSION_DESCRIPTION = """The {bot_name} bot has created a version of this dataset in the Parquet format in the {parquet_link} branch.

## What is Parquet?

Apache Parquet is a popular columnar storage format known for:

- reduced memory requirement,
- fast data retrieval and filtering,
- efficient storage.

**This is what powers the dataset viewer** on each dataset page and every dataset on the Hub can be accessed with the same code (you can use HF Datasets, ClickHouse, DuckDB, Pandas, PostgreSQL, or Polars, [up to you](https://huggingface.co/docs/dataset-viewer/parquet_process)).

You can learn more about the advantages associated with Parquet in the [documentation](https://huggingface.co/docs/dataset-viewer/parquet).

## How to access the Parquet version of the dataset?

You can access the Parquet version of the dataset by following this link: {parquet_link}

## What if my dataset was already in Parquet?

When the dataset is already in Parquet format, the data are not converted and the files in `refs/convert/parquet` are links to the original files.

## What should I do?

You don't need to do anything. The Parquet version of the dataset is available for you to use. Refer to the [documentation](https://huggingface.co/docs/dataset-viewer/parquet_process) for examples and code snippets on how to query the Parquet files with ClickHouse, DuckDB, Pandas or Polars.

If you have any questions or concerns, feel free to ask in the discussion below. You can also close the discussion if you don't have any questions."""


@dataclass
class ParquetCounters:
    datasets: int = 0
    new_discussions: int = 0
    dismissed_discussions: int = 0
    already_in_parquet_discussions: int = 0
    errors: int = 0


@dataclass
class Counters:
    parquet: ParquetCounters


def post_messages(
    hf_endpoint: str,
    bot_associated_user_name: Optional[str],
    bot_token: Optional[str],
    parquet_revision: str,
    skip_already_in_parquet: bool = True,
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
            skip_already_in_parquet=skip_already_in_parquet,
        )
    )


def post_messages_on_parquet_conversion(
    hf_endpoint: str,
    bot_associated_user_name: str,
    bot_token: str,
    parquet_revision: str,
    skip_already_in_parquet: bool = True,
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
            f" already had a discussion (open or closed), {counters.already_in_parquet_discussions} datasets"
            f" were already in Parquet (nothing to convert). {counters.errors} errors."
        )

    hf_api = HfApi(endpoint=hf_endpoint, token=bot_token)

    for dataset in datasets:
        counters.datasets += 1
        prefix = f"[{counters.datasets}/{len(datasets)}]"
        logging.info(f"{prefix} Processing dataset {dataset}")
        try:
            if skip_already_in_parquet and is_dataset_already_in_parquet(dataset):
                # the data is already stored as Parquet files on the Hub: the dataset viewer only
                # linked them in the Parquet revision, so there is nothing to announce to the user
                counters.already_in_parquet_discussions += 1
                logging.info(f"{prefix} [skipped] Dataset {dataset} is already in Parquet, no discussion opened")
            elif has_bot_discussion(
                hf_api=hf_api,
                dataset=dataset,
                bot_associated_user_name=bot_associated_user_name,
                bot_token=bot_token,
            ):
                # the bot has already opened a discussion for this dataset
                counters.dismissed_discussions += 1
                logging.info(f"{prefix} [dismissed] Dataset {dataset} already has a discussion, skipping")
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


def has_bot_discussion(hf_api: HfApi, dataset: str, bot_associated_user_name: str, bot_token: str) -> bool:
    """
    Tell if the bot has already opened a discussion (open or closed) for a dataset.

    Args:
        hf_api (`huggingface_hub.HfApi`): the HfApi to use, authenticated with the bot token.
        dataset (`str`): the dataset in question.
        bot_associated_user_name (`str`): the name of the Hub user associated with the bot.
        bot_token (`str`): the token of the bot.

    Returns:
        `bool`: True if the bot has already opened a discussion for this dataset.
    """
    try:
        next(
            hf_api.get_repo_discussions(
                repo_id=dataset, repo_type=REPO_TYPE_DATASET, token=bot_token, author=bot_associated_user_name
            )
        )
    except StopIteration:
        return False
    return True


def is_dataset_already_in_parquet(dataset: str) -> bool:
    """
    Tell if the data of a dataset is already stored as Parquet files on the Hub.

    In that case, the dataset viewer does not convert anything: it only links the original files
    in the Parquet revision (see "What if my dataset was already in Parquet?" in the discussion
    description), so there is nothing new to announce to the dataset owner, and we should not
    clutter their discussions with a "Conversion to Parquet" message.

    The information is read from the cached "dataset-info" response: its "builder_name" is
    "parquet" when a configuration points to Parquet files already on the Hub. Only datasets
    with at least one configuration, all of them being Parquet configurations, are considered as
    already in Parquet: as soon as one configuration has been converted, we keep opening a
    discussion.

    If the information is not available (no cached response, failed response, or unexpected
    content), the dataset is not considered as already in Parquet, and a discussion is opened
    (the historical behavior).

    Args:
        dataset (`str`): the dataset in question.

    Returns:
        `bool`: True if every configuration of the dataset is already in Parquet on the Hub.
    """
    try:
        response = get_response(kind=DATASET_INFO_KIND, dataset=dataset)
    except CachedArtifactNotFoundError:
        logging.debug(f"No cached '{DATASET_INFO_KIND}' response for {dataset}, opening a discussion anyway")
        return False
    if response["http_status"] != HTTPStatus.OK:
        logging.debug(f"Cached '{DATASET_INFO_KIND}' response for {dataset} is an error, opening a discussion anyway")
        return False
    config_infos = response["content"].get("dataset_info")
    if not isinstance(config_infos, dict):
        logging.debug(f"Unexpected '{DATASET_INFO_KIND}' content for {dataset}, opening a discussion anyway")
        return False
    builder_names: list[Any] = [
        config_info.get("builder_name") for config_info in config_infos.values() if isinstance(config_info, dict)
    ]
    return bool(builder_names) and all(builder_name == PARQUET_BUILDER_NAME for builder_name in builder_names)


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
