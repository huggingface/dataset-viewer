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
    DatasetBuilderScriptConfigNoSplitsError,
    DatasetNotFoundError,
    ConfigNotFoundError,
    SplitError,
    SplitNotImplementedError,
)

# TODO: log the traces on every caught exception


def get_configs(dataset: str) -> List[str]:
    try:
        module_path, *_ = prepare_module(dataset, dataset=True)
        builder_cls = import_main_class(module_path, dataset=True)
    except FileNotFoundError as err:
        raise DatasetNotFoundError(dataset=dataset)
    except (ModuleNotFoundError):
        raise DatasetBuilderScriptError(dataset=dataset)

    configs = [c.name for c in builder_cls.BUILDER_CONFIGS] or [None]
    logging.debug(f"The dataset builder has {len(configs)} configs: {configs}")
    return {"dataset": dataset, "configs": configs}


def get_splits(dataset: str, config: str) -> List[str]:
    try:
        builder = load_dataset_builder(dataset, name=config)
    except ValueError as err:
        message = str(err)
        if message.startswith(f"BuilderConfig {config} not found"):
            raise ConfigNotFoundError(dataset=dataset, config=config)
        elif message.startswith(f"Config name is missing."):
            raise ConfigNotFoundError(dataset=dataset, config=config)
        else:
            raise
    except (ModuleNotFoundError, RuntimeError, TypeError):
        raise DatasetBuilderScriptError(dataset=dataset)

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
                dataset=dataset, config=config
            )
    else:
        splits = list(builder.info.splits.keys())
    return {"dataset": dataset, "config": config, "splits": splits}


def extract_rows(dataset: str, config: str, split: str, num_rows: int):
    logging.debug(
        f"asked for {num_rows} first rows of dataset {dataset} - {config} - {split}"
    )

    try:
        iterable_dataset: IterableDataset = load_dataset(
            dataset, name=config, split=split, streaming=True
        )
    except FileNotFoundError as err:
        raise DatasetNotFoundError(dataset=dataset)
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
            dataset=dataset,
            config=config,
            split=split,
            extension=extension,
        )
    except ValueError as err:
        message = str(err)
        if message.startswith(f"BuilderConfig {config} not found"):
            raise ConfigNotFoundError(dataset=dataset, config=config)
        elif message.startswith(f"Config name is missing."):
            raise ConfigNotFoundError(dataset=dataset, config=config)
        elif message.startswith(f'Unknown split "{split}".') or message.startswith(
            f"Bad split: {split}."
        ):
            raise SplitError(dataset=dataset, config=config, split=split)
        else:
            raise
    except (ModuleNotFoundError, RuntimeError, TypeError):
        raise DatasetBuilderScriptError(dataset=dataset)

    rows = list(iterable_dataset.take(num_rows))
    if len(rows) != num_rows:
        logging.warning(
            f"could not read all the required rows ({len(rows)} / {num_rows}) from dataset {dataset} - {config} - {split}"
        )

    return {
        "dataset": dataset,
        "config": config,
        "split": split,
        "rows": rows,
    }
