import typer
from datasets import list_datasets

from datasets_preview_backend.serialize import serialize_dataset_name


def main(filename: str) -> None:
    dataset_names = list_datasets(with_community_datasets=True)  # type: ignore
    # replace '/' in namespaced dataset names
    serialized_dataset_names = [serialize_dataset_name(dataset_name) for dataset_name in dataset_names]
    with open(filename, "w") as f:
        for serialized_dataset_name in serialized_dataset_names:
            f.write("%s\n" % serialized_dataset_name)


if __name__ == "__main__":
    typer.run(main)
