import requests
import typer

from datasets_preview_backend.serialize import serialize_params

# TODO: use env vars + add an env var for the scheme (http/https)
ENDPOINT = "http://localhost:8000/"


def main(filename: str) -> None:
    r = requests.get(f"{ENDPOINT}datasets")
    r.raise_for_status()
    d = r.json()
    dataset_names = d["datasets"]
    # replace '/' in namespaced dataset names
    serialized_dataset_names = [serialize_params({"dataset": dataset_name}) for dataset_name in dataset_names]
    # tmp:
    # serialized_dataset_names = serialized_dataset_names[0:15]
    with open(filename, "w") as f:
        for serialized_dataset_name in serialized_dataset_names:
            f.write("%s\n" % serialized_dataset_name)


if __name__ == "__main__":
    typer.run(main)
