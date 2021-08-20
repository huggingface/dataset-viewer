import re
import logging

from typing import List

from datasets import (
    IterableDataset,
    load_dataset,
)

from datasets_preview_backend.exceptions import (
    Status400Error,
    Status404Error,
)


def extract_rows(dataset: str, config: str, split: str, num_rows: int):
    logging.debug(
        f"asked for {num_rows} first rows of dataset {dataset} - {config} - {split}"
    )

    try:
        iterable_dataset: IterableDataset = load_dataset(
            dataset, name=config, split=split, streaming=True
        )
        rows = list(iterable_dataset.take(num_rows))
    except FileNotFoundError as err:
        raise Status404Error(
            "The split for the dataset config could not be found."
        ) from err
    except NotImplementedError as err:
        # TODO: check what has changed once https://github.com/huggingface/datasets/pull/2662 is merged
        try:
            regex = re.compile(
                r"Extraction protocol for file at .*?((\.\w+)?\.\w+)* is not implemented yet"
            )
            extension = regex.match(str(err)).group(1)
        except:
            raise Status400Error(
                "The rows could not be extracted from the split of the dataset config."
            ) from err
        else:
            raise Status400Error(
                f"The rows could not be extracted from the split of the dataset config because extension {extension} is not supported."
            ) from err
    except ValueError as err:
        if (
            str(err).startswith(f"BuilderConfig {config} not found.")
            or str(err).startswith(f"Config name is missing.")
            or str(err).startswith(f"Bad split")
        ):
            raise Status404Error(
                "The dataset config could not be found.") from err
        else:
            raise Status400Error(
                "The rows could not be extracted from the split of the dataset config."
            ) from err
    except Exception as err:
        raise Status400Error(
            "The rows could not be extracted from the split of the dataset config."
        ) from err

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
