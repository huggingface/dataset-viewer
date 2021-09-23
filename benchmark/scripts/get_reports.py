import json

import typer

from datasets_preview_backend.reports import get_report_dict
from datasets_preview_backend.serialize import deserialize_params


def main(url: str, endpoint: str, serialized_params: str, filename: str) -> None:
    params = deserialize_params(serialized_params)
    report_dict = get_report_dict(url=url, endpoint=endpoint, params=params)
    with open(filename, "w") as f:
        json.dump(report_dict, f)


if __name__ == "__main__":
    typer.run(main)
