import json

import typer

from datasets_preview_backend.reports import get_request_report
from datasets_preview_backend.serialize import deserialize_params


def main(url: str, endpoint: str, serialized_params: str, filename: str) -> None:
    params = deserialize_params(serialized_params)
    report = get_request_report(url=url, endpoint=endpoint, params=params)
    with open(filename, "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    typer.run(main)
