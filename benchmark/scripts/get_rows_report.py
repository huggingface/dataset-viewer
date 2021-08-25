import json
import logging
import time

import typer
from datasets import disable_progress_bar
from serialize import deserialize_split_name

from datasets_preview_backend.queries.rows import extract_rows

# remove any logs
logging.disable(logging.CRITICAL)
disable_progress_bar()


def get_rows_report(dataset: str, config: str, split: str):
    num_rows = 10
    try:
        t = time.process_time()
        rows = extract_rows(dataset, config, split, num_rows)["rows"]
        return {
            "dataset": dataset,
            "config": config,
            "split": split,
            "rows_length": len(rows),
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
            "split": split,
            "success": False,
            "exception": type(err).__name__,
            "message": str(err),
            "cause": type(err.__cause__).__name__,
            "cause_message": str(err.__cause__),
            "elapsed_seconds": time.process_time() - t,
        }


def main(serialized_split_name: str, filename: str):
    dataset, config, split = deserialize_split_name(serialized_split_name)
    report = get_rows_report(dataset, config, split)
    with open(filename, "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    typer.run(main)
