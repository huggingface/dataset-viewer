import json
from typing import List

import typer

from datasets_preview_backend.reports import ConfigsReportDict
from datasets_preview_backend.serialize import serialize_config_name


def main(get_configs_reports_filename: str, output: str) -> None:
    with open(get_configs_reports_filename) as f:
        get_configs_reports: List[ConfigsReportDict] = json.load(f)

    serialized_config_names = []
    for get_configs_report in get_configs_reports:
        if get_configs_report["result"] is not None:
            dataset = get_configs_report["args"]["dataset"]
            for config in get_configs_report["result"]["configs"]:
                # replace special characters datasets and configs names
                serialized_config_names.append(serialize_config_name(dataset, config))

    with open(output, "w") as f:
        for serialized_config_name in serialized_config_names:
            f.write("%s\n" % serialized_config_name)


if __name__ == "__main__":
    typer.run(main)
