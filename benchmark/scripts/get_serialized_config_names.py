import json
from typing import List

import typer

from datasets_preview_backend.reports import CommonReportDict
from datasets_preview_backend.serialize import serialize_params


def main(get_configs_reports_filename: str, output: str) -> None:
    with open(get_configs_reports_filename) as f:
        get_configs_reports: List[CommonReportDict] = json.load(f)

    serialized_config_names = []
    for get_configs_report in get_configs_reports:
        if (
            get_configs_report is None
            or get_configs_report["params"] is None
            or "dataset" not in get_configs_report["params"]
            or get_configs_report["result"] is None
            or "configs" not in get_configs_report["result"]
        ):
            continue
        dataset = get_configs_report["params"]["dataset"]
        for config in get_configs_report["result"]["configs"]:
            params = {"dataset": dataset, "config": config}
            # replace special characters datasets and configs names
            serialized_config_names.append(serialize_params(params))

    with open(output, "w") as f:
        for serialized_config_name in serialized_config_names:
            f.write("%s\n" % serialized_config_name)


if __name__ == "__main__":
    typer.run(main)
