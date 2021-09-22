import json
import logging
import time

import requests
import typer

from datasets_preview_backend.reports import SplitsReport
from datasets_preview_backend.serialize import deserialize_config_name

# remove any logs
logging.disable(logging.CRITICAL)


# TODO: use env vars + add an env var for the scheme (http/https)
ENDPOINT = "http://localhost:8000/"


def get_splits_report(dataset: str, config: str) -> SplitsReport:
    t = time.process_time()
    r = requests.get(f"{ENDPOINT}splits?dataset={dataset}&config={config}")
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
    return SplitsReport(
        args={"dataset": dataset, "config": config}, response=response, error=error, elapsed_seconds=elapsed_seconds
    )


def main(serialized_config_name: str, filename: str) -> None:
    dataset, config = deserialize_config_name(serialized_config_name)
    report = get_splits_report(dataset, config)
    with open(filename, "w") as f:
        json.dump(report.to_dict(), f)


if __name__ == "__main__":
    typer.run(main)
