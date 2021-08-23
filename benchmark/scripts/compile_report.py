import json

import time
import typer


def main(
    get_info_reports_filename: str,
    get_configs_reports_filename: str,
    get_splits_reports_filename: str,
    get_rows_reports_filename: str,
    output: str,
):
    with open(get_info_reports_filename) as f:
        get_info_reports = json.load(f)
    with open(get_configs_reports_filename) as f:
        get_configs_reports = json.load(f)
    with open(get_splits_reports_filename) as f:
        get_splits_reports = json.load(f)
    with open(get_rows_reports_filename) as f:
        get_rows_reports = json.load(f)
    time_string = time.strftime("%Y%m%d-%H%M%S")
    report = {
        "info_reports": get_info_reports,
        "configs_reports": get_configs_reports,
        "splits_reports": get_splits_reports,
        "rows_reports": get_rows_reports,
        "created_at": time_string,
    }
    with open(output, "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    typer.run(main)
