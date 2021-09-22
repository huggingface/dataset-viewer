import json
from typing import List

import typer

from datasets_preview_backend.reports import SplitsReportDict
from datasets_preview_backend.serialize import serialize_split_name


def main(get_splits_reports_filename: str, output: str) -> None:
    with open(get_splits_reports_filename) as f:
        get_splits_reports: List[SplitsReportDict] = json.load(f)

    serialized_split_names = []
    for get_splits_report in get_splits_reports:
        if get_splits_report["result"] is not None:
            dataset = get_splits_report["args"]["dataset"]
            config = get_splits_report["args"]["config"]
            for split in get_splits_report["result"]["splits"]:
                # replace special characters datasets and configs names
                serialized_split_names.append(serialize_split_name(dataset, config, split))

    with open(output, "w") as f:
        for serialized_split_name in serialized_split_names:
            f.write("%s\n" % serialized_split_name)


if __name__ == "__main__":
    typer.run(main)
