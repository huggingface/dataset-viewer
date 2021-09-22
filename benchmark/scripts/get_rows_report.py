import json
import logging
import time

import requests
import typer

from datasets_preview_backend.reports import RowsReport
from datasets_preview_backend.serialize import deserialize_split_name

# remove any logs
logging.disable(logging.CRITICAL)


# TODO: use env vars + add an env var for the scheme (http/https)
ENDPOINT = "http://localhost:8000/"

NUM_ROWS = 100


def get_rows_report(dataset: str, config: str, split: str) -> RowsReport:
    t = time.process_time()
    r = requests.get(f"{ENDPOINT}rows?dataset={dataset}&config={config}&split={split}&rows={NUM_ROWS}")
    try:
        r.raise_for_status()
        response = r.json()
        error = None
    except Exception as err:
        response = None
        if r.status_code in [400, 404]:
            # these error code are managed and return a json we can parse
            error = r.json()
        else:
            error = {
                "exception": type(err).__name__,
                "message": str(err),
                "cause": type(err.__cause__).__name__,
                "cause_message": str(err.__cause__),
                "status_code": r.status_code,
            }
    elapsed_seconds = time.process_time() - t
    return RowsReport(
        args={"dataset": dataset, "config": config, "split": split, "num_rows": NUM_ROWS},
        response=response,
        error=error,
        elapsed_seconds=elapsed_seconds,
    )


def main(serialized_split_name: str, filename: str) -> None:
    dataset, config, split = deserialize_split_name(serialized_split_name)
    report = get_rows_report(dataset, config, split)
    with open(filename, "w") as f:
        json.dump(report.to_dict(), f)


if __name__ == "__main__":
    typer.run(main)
