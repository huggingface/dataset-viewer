import json
import logging
import time

import typer
from datasets import disable_progress_bar
from serialize import deserialize_config_name

from datasets_preview_backend.queries.splits import get_splits

# remove any logs
logging.disable(logging.CRITICAL)
disable_progress_bar()


def get_splits_report(dataset: str, config: str):
    try:
        t = time.process_time()
        splits = get_splits(dataset, config)["splits"]
        return {
            "dataset": dataset,
            "config": config,
            "splits": list(splits),
            "success": True,
            "exception": None,
            "message": None,
            "cause": None,
            "cause_message": None,
            "elapsed_seconds": time.process_time() - t,
        }
    except Exception as err:
        return {
            "dataset": dataset,
            "config": config,
            "splits": [],
            "success": False,
            "exception": type(err).__name__,
            "message": str(err),
            "cause": type(err.__cause__).__name__,
            "cause_message": str(err.__cause__),
            "elapsed_seconds": time.process_time() - t,
        }


def main(serialized_config_name: str, filename: str):
    dataset, config = deserialize_config_name(serialized_config_name)
    splits = get_splits_report(dataset, config)
    with open(filename, "w") as f:
        json.dump(splits, f)


if __name__ == "__main__":
    typer.run(main)
