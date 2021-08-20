from datasets_preview_backend.queries.rows import extract_rows
from datasets_preview_backend.queries.splits import get_splits
from datasets_preview_backend.queries.configs import get_configs
from datasets_preview_backend.queries.info import get_info
from datasets import list_datasets, disable_progress_bar
import json
import time
import logging
import concurrent.futures

# remove any logs
logging.disable(logging.CRITICAL)
disable_progress_bar()


def get_info_report(dataset: str):
    try:
        info = get_info(dataset)["info"]
        return {
            "dataset": dataset,
            "info": info,
            "success": True,
            "exception": None,
            "message": None,
            "cause": None,
            "cause_message": None,
        }
    except Exception as err:
        return {
            "dataset": dataset,
            "info": None,
            "success": False,
            "exception": type(err).__name__,
            "message": str(err),
            "cause": type(err.__cause__).__name__,
            "cause_message": str(err.__cause__),
        }


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


def process_map(fun, iterator, max_workers):
    # We can use a with statement to ensure threads are cleaned up promptly
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start the load operations and mark each future with its URL
        future_to_item = {executor.submit(fun, **item): item for item in iterator}
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            print(item)
            try:
                result = future.result()
            except Exception as exc:
                print("%r generated an exception: %s" % (item, exc))
            else:
                results.append(result)
    return results


def export_all_datasets_exceptions():
    max_workers = 5
    datasets = list_datasets(with_community_datasets=True)
    datasets_iterator = [{"dataset": dataset} for dataset in datasets]

    # print("Get info for all the datasets")
    info_reports = process_map(
        get_info_report, datasets_iterator, max_workers=max_workers
    )

    print("Get config names for all the datasets")
    configs_reports = process_map(
        get_configs_report, datasets_iterator, max_workers=max_workers
    )

    print("Get split names for all the pairs (dataset, config)")
    configs_iterator = []
    for report in configs_reports:
        for config in report["configs"]:
            # reports with an exception will not contribute to the lists since configs is empty
            configs_iterator.append({"dataset": report["dataset"], "config": config})
    splits_reports = process_map(
        get_splits_report,
        configs_iterator,
        max_workers=max_workers,
    )

    print("Get rows extract for all the tuples (dataset, config, split)")
    splits_iterator = []
    for report in splits_reports:
        for split in report["splits"]:
            # reports with an exception will not contribute to the lists since splits is empty
            splits_iterator.append(
                {
                    "dataset": report["dataset"],
                    "config": report["config"],
                    "split": split,
                }
            )
    rows_reports = process_map(
        get_rows_report,
        splits_iterator,
        max_workers=max_workers,
    )

    results = {
        "info_reports": info_reports,
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
