import re
import logging

from typing import List

from datasets import (
    IterableDataset,
    load_dataset,
)

from datasets_preview_backend.exceptions import (
    DatasetBuilderScriptError,
    DatasetNotFoundError,
    ConfigNotFoundError,
    SplitError,
    SplitNotImplementedError,
)

# TODO: log the traces on every caught exception


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
