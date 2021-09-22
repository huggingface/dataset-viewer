import json
import logging
import time

import requests
import typer

from datasets_preview_backend.reports import InfoReport
from datasets_preview_backend.serialize import deserialize_dataset_name

# remove any logs
logging.disable(logging.CRITICAL)

# TODO: use env vars + add an env var for the scheme (http/https)
ENDPOINT = "http://localhost:8000/"


def get_info_report(dataset: str) -> InfoReport:
    t = time.process_time()
    r = requests.get(f"{ENDPOINT}info?dataset={dataset}")
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
    return InfoReport(args={"dataset": dataset}, response=response, error=error, elapsed_seconds=elapsed_seconds)


def main(serialized_dataset_name: str, filename: str) -> None:
    report = get_info_report(deserialize_dataset_name(serialized_dataset_name))
    with open(filename, "w") as f:
        json.dump(report.to_dict(), f)


if __name__ == "__main__":
    typer.run(main)
