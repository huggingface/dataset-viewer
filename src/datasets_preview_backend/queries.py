import re
import logging

from typing import List

from datasets import (
    IterableDataset,
    load_dataset,
    load_dataset_builder,
    prepare_module,
    import_main_class,
)
from datasets.utils.streaming_download_manager import StreamingDownloadManager

from datasets_preview_backend.exceptions import (
    DatasetBuilderScriptError,
    DatasetBuilderScriptConfigError,
    DatasetBuilderScriptConfigNoSplitsError,
    DatasetNotFoundError,
    ConfigNotFoundError,
    SplitError,
    SplitNotImplementedError,
)


def get_dataset_config_names(dataset_id: str) -> List[str]:
    try:
        module_path, *_ = prepare_module(dataset_id, dataset=True)
        builder_cls = import_main_class(module_path, dataset=True)
    except FileNotFoundError as err:
        raise DatasetNotFoundError(dataset_id=dataset_id)
    except (ModuleNotFoundError):
        raise DatasetBuilderScriptError(dataset_id=dataset_id)

    config_names = [c.name for c in builder_cls.BUILDER_CONFIGS] or [None]
    logging.debug(
        f"The dataset builder has {len(config_names)} configs: {config_names}"
    )
    return config_names


def get_config_splits(dataset_id: str, config_name: str) -> List[str]:
    try:
        builder = load_dataset_builder(dataset_id, name=config_name)
    except ValueError as err:
        message = str(err)
        if message.startswith(f"BuilderConfig {config_name} not found"):
            raise ConfigNotFoundError(dataset_id=dataset_id, config_name=config_name)
        else:
            raise
    except (ModuleNotFoundError, RuntimeError, TypeError):
        raise DatasetBuilderScriptConfigError(
            dataset_id=dataset_id, config_name=config_name
        )

    if builder.info.splits is None:
        # try to get them from _split_generators
        try:
            splits = [
                split_generator.name
                for split_generator in builder._split_generators(
                    StreamingDownloadManager(base_path=builder.base_path)
                )
            ]
        except:
            raise DatasetBuilderScriptConfigNoSplitsError(
                dataset_id=dataset_id, config_name=config_name
            )
    else:
        splits = list(builder.info.splits.keys())
    return splits


def extract_split_rows(dataset_id: str, config_name: str, split: str, num_rows: int):
    logging.debug(
        f"asked for {num_rows} first rows of dataset {dataset_id} - {config_name} - {split}"
    )

    try:
        dataset: IterableDataset = load_dataset(
            dataset_id, name=config_name, split=split, streaming=True
        )
    except NotImplementedError as err:
        # TODO: check what has changed once https://github.com/huggingface/datasets/pull/2662 is merged
        try:
            regex = re.compile(
                r"Extraction protocol for file at .*?((\.\w+)?\.\w+)* is not implemented yet"
            )
            extension = regex.match(str(err)).group(1)
        except:
            extension = None
        raise SplitNotImplementedError(
            dataset_id=dataset_id,
            config_name=config_name,
            split=split,
            extension=extension,
        )
    except ValueError as err:
        message = str(err)
        if message.startswith(f"BuilderConfig {config_name} not found"):
            raise ConfigNotFoundError(dataset_id=dataset_id, config_name=config_name)
        elif message.startswith(f'Unknown split "{split}".') or message.startswith(
            f"Bad split: {split}."
        ):
            raise SplitError(
                dataset_id=dataset_id, config_name=config_name, split=split
            )
        else:
            raise

    rows = list(dataset.take(num_rows))
    if len(rows) != num_rows:
        logging.warning(
            f"could not read all the required rows ({len(rows)} / {num_rows}) from dataset {dataset_id} - {config_name} - {split}"
        )

    return {
        "dataset_id": dataset_id,
        "config_name": config_name,
        "split": split,
        "rows": rows,
    }


def extract_config_rows(dataset_id: str, config_name: str, num_rows: int):
    logging.debug(
        f"asked for {num_rows} first rows of dataset {dataset_id} - {config_name}"
    )

    splits = get_config_splits(dataset_id, config_name)

    return {
        "dataset_id": dataset_id,
        "config_name": config_name,
        "splits": {
            split: extract_split_rows(dataset_id, config_name, split, num_rows)
            for split in splits
        },
    }


def extract_dataset_rows(dataset_id: str, num_rows: int):
    logging.debug(f"asked for {num_rows} first rows of dataset {dataset_id}")

    config_names = get_dataset_config_names(dataset_id)

    return {
        "dataset_id": dataset_id,
        "configs": {
            config_name: extract_config_rows(dataset_id, config_name, num_rows)
            for config_name in config_names
        },
    }
