import json

import typer

from datasets_preview_backend.serialize import serialize_split_name


def main(get_splits_reports_filename: str, output: str) -> None:
    with open(get_splits_reports_filename) as f:
        get_splits_reports = json.load(f)

    # replace '/' in namespaced dataset names
    serialized_split_names = []
    for get_splits_report in get_splits_reports:
        dataset = get_splits_report["dataset"]
        config = get_splits_report["config"]
        for split in get_splits_report["splits"]:
            serialized_split_names.append(serialize_split_name(dataset, config, split))

    with open(output, "w") as f:
        for serialized_split_name in serialized_split_names:
            f.write("%s\n" % serialized_split_name)


if __name__ == "__main__":
    typer.run(main)
