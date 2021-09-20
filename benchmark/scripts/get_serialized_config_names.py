import json

import typer

from datasets_preview_backend.serialize import serialize_config_name


def main(get_configs_reports_filename: str, output: str) -> None:
    with open(get_configs_reports_filename) as f:
        get_configs_reports = json.load(f)

    # replace '/' in namespaced dataset names
    serialized_config_names = []
    for get_configs_report in get_configs_reports:
        dataset = get_configs_report["dataset"]
        for config in get_configs_report["configs"]:
            serialized_config_names.append(serialize_config_name(dataset, config))

    with open(output, "w") as f:
        for serialized_config_name in serialized_config_names:
            f.write("%s\n" % serialized_config_name)


if __name__ == "__main__":
    typer.run(main)
