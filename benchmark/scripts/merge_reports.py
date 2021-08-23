import json
import typer
import os


def main(
    basenames_filename: str,
    reports_dir: str,
    output: str,
):
    reports = []
    with open(basenames_filename) as f:
        basenames = f.read().splitlines()
        for basename in basenames:
            filename = os.path.join(reports_dir, basename + ".json")
            with open(filename) as f2:
                reports.append(json.load(f2))
    with open(output, "w") as f:
        json.dump(reports, f)


if __name__ == "__main__":
    typer.run(main)
