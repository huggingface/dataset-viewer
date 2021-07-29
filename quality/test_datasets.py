import json
import time
from tqdm import tqdm

from datasets import list_datasets

from datasets_preview_backend.main import extract_dataset_rows


def export_all_datasets_exceptions():
    num_rows = 10
    dataset_ids = list_datasets(with_community_datasets=True)

    results = []

    for dataset_id in tqdm(dataset_ids):

        success = False
        try:
            extract = extract_dataset_rows(dataset_id, num_rows)
            exception = ""
            config_names = extract["configs"].keys()
            split_names = set()
            for config_name, config in extract["configs"].items():
                for split_name, split in config["splits"].items():
                    split_names.add(split_name)
                    if len(split["rows"]) != num_rows:
                        raise ValueError(
                            f"{len(split['rows'])} rows instead of {num_rows} in {config_name} - {split_name}"
                        )
            success = True
            message = ""
        except Exception as err:
            exception = str(type(err).__name__)
            message = str(err)
            config_names = []
            split_names = []
        results.append(
            {
                "dataset_id": dataset_id,
                "success": success,
                "exception": exception,
                "message": message,
                "all_config_names": list(config_names),
                "all_split_names": list(split_names),
            }
        )

    time_string = time.strftime("%Y%m%d-%H%M%S")
    filename = f"/tmp/datasets-{time_string}.json"
    with open(filename, "w") as outfile:
        json.dump(results, outfile, indent=2)
    print(f"report has been written at {filename}")


if __name__ == "__main__":
    export_all_datasets_exceptions()
