import json
import logging
import time

import typer
from datasets.utils.tqdm_utils import disable_progress_bar

from datasets_preview_backend.queries.configs import get_configs
from datasets_preview_backend.serialize import deserialize_dataset_name
from datasets_preview_backend.types import ConfigsReport

# remove any logs
logging.disable(logging.CRITICAL)
disable_progress_bar()  # type: ignore


def get_configs_report(dataset: str) -> ConfigsReport:
    try:
        t = time.process_time()
        configs = get_configs(dataset)["configs"]
        return {
            "dataset": dataset,
            "configs": list(configs),
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
            "configs": [],
            "success": False,
            "exception": type(err).__name__,
            "message": str(err),
            "cause": type(err.__cause__).__name__,
            "cause_message": str(err.__cause__),
            "elapsed_seconds": time.process_time() - t,
        }


def main(serialized_dataset_name: str, filename: str) -> None:
    report = get_configs_report(deserialize_dataset_name(serialized_dataset_name))
    with open(filename, "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    typer.run(main)
