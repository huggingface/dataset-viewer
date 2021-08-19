from datasets import list_datasets, disable_progress_bar
import json
import time
from tqdm.contrib.concurrent import process_map
import logging

# remove any logs
logging.disable(logging.CRITICAL)
disable_progress_bar()

from datasets_preview_backend.queries.configs import get_configs
from datasets_preview_backend.queries.splits import get_splits
from datasets_preview_backend.queries.rows import extract_rows


def get_configs_report(dataset: str):
    try:
        configs = get_configs(dataset)["configs"]
        return {
            "dataset": dataset,
            "configs": list(configs),
            "success": True,
            "exception": None,
            "message": None,
            "cause": None,
            "cause_message": None,
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
        }


def get_splits_report(dataset: str, config: str):
    try:
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
        }


def get_rows_report(dataset: str, config: str, split: str):
    num_rows = 10
    try:
        rows = extract_rows(dataset, config, split, num_rows)["rows"]
        return {
            "dataset": dataset,
            "config": config,
            "split": split,
            "row_length": len(rows),
            "success": True,
            "exception": None,
            "message": None,
            "cause": None,
            "cause_message": None,
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
        }


def export_all_datasets_exceptions():
    chunksize = 5
    datasets = list_datasets(with_community_datasets=True)

    print("Get config names for all the datasets")
    configs_reports = process_map(get_configs_report, datasets, chunksize=chunksize)

    print("Get split names for all the pairs (dataset, config)")
    splits_datasets = []
    splits_configs = []
    for report in configs_reports:
        for config in report["configs"]:
            # reports with an exception will not contribute to the lists since configs is empty
            splits_datasets.append(report["dataset"])
            splits_configs.append(config)
    splits_reports = process_map(
        get_splits_report,
        splits_datasets,
        splits_configs,
        chunksize=chunksize,
    )

    print("Get rows extract for all the tuples (dataset, config, split)")
    rows_datasets = []
    rows_configs = []
    rows_splits = []
    for report in splits_reports:
        for split in report["splits"]:
            # reports with an exception will not contribute to the lists since splits is empty
            rows_datasets.append(report["dataset"])
            rows_configs.append(report["config"])
            rows_splits.append(split)
    rows_reports = process_map(
        get_rows_report,
        rows_datasets,
        rows_configs,
        rows_splits,
        chunksize=chunksize,
    )

    results = {
        "configs_reports": configs_reports,
        "splits_reports": splits_reports,
        "rows_reports": rows_reports,
    }

    time_string = time.strftime("%Y%m%d-%H%M%S")
    filename = f"/tmp/datasets-{time_string}.json"
    with open(filename, "w") as outfile:
        json.dump(results, outfile, indent=2)
    print(f"report has been written at {filename}")


if __name__ == "__main__":
    export_all_datasets_exceptions()
