import json
import logging
import time

import typer
from datasets.utils.tqdm_utils import disable_progress_bar

from datasets_preview_backend.queries.splits import get_splits
from datasets_preview_backend.serialize import deserialize_config_name
from datasets_preview_backend.types import SplitsReport

# remove any logs
logging.disable(logging.CRITICAL)
disable_progress_bar()  # type: ignore


def get_splits_report(dataset: str, config: str) -> SplitsReport:
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


def main(serialized_config_name: str, filename: str) -> None:
    dataset, config = deserialize_config_name(serialized_config_name)
    report = get_splits_report(dataset, config)
    with open(filename, "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    typer.run(main)
