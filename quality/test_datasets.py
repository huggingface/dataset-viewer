from datasets import list_datasets
import json
import time
from tqdm.contrib.concurrent import process_map


from datasets_preview_backend.queries import (
    get_config_names,
    get_splits,
    extract_rows,
)


def get_config_names_report(dataset_id: str):
    try:
        config_names = get_config_names(dataset_id)
        return {
            "dataset_id": dataset_id,
            "config_names": list(config_names),
            "success": True,
            "exception": None,
            "message": None,
        }
    except Exception as err:
        return {
            "dataset_id": dataset_id,
            "config_names": [],
            "success": False,
            "exception": str(type(err).__name__),
            "message": str(err),
        }


def get_split_names_report(dataset_id: str, config_name: str):
    try:
        split_names = get_splits(dataset_id, config_name)
        return {
            "dataset_id": dataset_id,
            "config_name": config_name,
            "split_names": list(split_names),
            "success": True,
            "exception": None,
            "message": None,
        }
    except Exception as err:
        return {
            "dataset_id": dataset_id,
            "config_name": config_name,
            "split_names": [],
            "success": False,
            "exception": str(type(err).__name__),
            "message": str(err),
        }


def get_rows_report(dataset_id: str, config_name: str, split_name: str):
    num_rows = 10
    try:
        extract = extract_rows(dataset_id, config_name, split_name, num_rows)
        if len(extract["rows"]) != num_rows:
            raise ValueError(f"{len(extract['rows'])} rows instead of {num_rows}")
        return {
            "dataset_id": dataset_id,
            "config_name": config_name,
            "split_name": split_name,
            "success": True,
            "exception": None,
            "message": None,
        }
    except Exception as err:
        return {
            "dataset_id": dataset_id,
            "config_name": config_name,
            "split_name": split_name,
            "success": False,
            "exception": str(type(err).__name__),
            "message": str(err),
        }


def export_all_datasets_exceptions():
    dataset_ids = list_datasets(with_community_datasets=True)

    print("Get config names for all the datasets")
    config_names_reports = process_map(
        get_config_names_report, dataset_ids, chunksize=20
    )

    print("Get split names for all the pairs (dataset_id, config_name)")
    split_names_dataset_ids = []
    split_names_config_names = []
    for report in config_names_reports:
        for config_name in report["config_names"]:
            # reports with an exception will not contribute to the lists since config_names is empty
            split_names_dataset_ids.append(report["dataset_id"])
            split_names_config_names.append(config_name)
    split_names_reports = process_map(
        get_split_names_report,
        split_names_dataset_ids,
        split_names_config_names,
        chunksize=20,
    )

    print("Get rows extract for all the tuples (dataset_id, config_name, split_name)")
    rows_dataset_ids = []
    rows_config_names = []
    rows_split_names = []
    for report in split_names_reports:
        for split_name in report["split_names"]:
            # reports with an exception will not contribute to the lists since split_names is empty
            rows_dataset_ids.append(report["dataset_id"])
            rows_config_names.append(report["config_name"])
            rows_split_names.append(split_name)
    rows_reports = process_map(
        get_rows_report,
        rows_dataset_ids,
        rows_config_names,
        rows_split_names,
        chunksize=20,
    )

    results = {
        "config_names_reports": config_names_reports,
        "split_names_reports": split_names_reports,
        "rows_reports": rows_reports,
    }

    time_string = time.strftime("%Y%m%d-%H%M%S")
    filename = f"/tmp/datasets-{time_string}.json"
    with open(filename, "w") as outfile:
        json.dump(results, outfile, indent=2)
    print(f"report has been written at {filename}")


if __name__ == "__main__":
    export_all_datasets_exceptions()
