import json
import time
from tqdm import tqdm

from datasets import list_datasets

from datasets_preview_backend.main import get_dataset_extract


def export_all_datasets_exceptions():
    num_rows = 100
    dataset_ids = list_datasets(with_community_datasets=True)

    results = []

    for dataset_id in tqdm(dataset_ids[0:2]):

        success = False
        try:
            extract = get_dataset_extract(dataset_id, num_rows)
            exception = ""
            if len(extract) != num_rows:
                raise f"{len(extract)} rows instead of {num_rows}"
            success = True
            message = ""
        except Exception as err:
            exception = str(type(err).__name__)
            message = str(err)
        results.append(
            {
                "dataset_id": dataset_id,
                "success": success,
                "exception": exception,
                "message": message,
            }
        )

    time_string = time.strftime("%Y%m%d-%H%M%S")
    filename = f"/tmp/datasets-{time_string}.json"
    with open(filename, "w") as outfile:
        json.dump(results, outfile, indent=2)
    print(f"report has been written at {filename}")


if __name__ == "__main__":
    export_all_datasets_exceptions()
