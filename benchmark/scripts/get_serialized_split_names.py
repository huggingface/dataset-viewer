import json
from typing import List

import typer

from datasets_preview_backend.reports import RequestReportDict
from datasets_preview_backend.serialize import serialize_params


def main(get_splits_reports_filename: str, output: str) -> None:
    with open(get_splits_reports_filename) as f:
        get_splits_reports: List[RequestReportDict] = json.load(f)

    serialized_split_names = []
    for get_splits_report in get_splits_reports:
        if (
            get_splits_report is None
            or get_splits_report["params"] is None
            or "dataset" not in get_splits_report["params"]
            or "config" not in get_splits_report["params"]
            or get_splits_report["result"] is None
            or "splits" not in get_splits_report["result"]
        ):
            continue
        dataset = get_splits_report["params"]["dataset"]
        config = get_splits_report["params"]["config"]
        for split in get_splits_report["result"]["splits"]:
            params = {"dataset": dataset, "config": config, "split": split}
            # replace special characters datasets and configs names
            serialized_split_names.append(serialize_params(params))

    with open(output, "w") as f:
        for serialized_split_name in serialized_split_names:
            f.write("%s\n" % serialized_split_name)


if __name__ == "__main__":
    typer.run(main)
