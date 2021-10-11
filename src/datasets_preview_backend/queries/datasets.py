from typing import List, TypedDict

from datasets_preview_backend.dataset_entries import get_dataset_names


class DatasetItem(TypedDict):
    dataset: str


class DatasetsContent(TypedDict):
    datasets: List[DatasetItem]


def get_datasets() -> DatasetsContent:
    return {"datasets": [{"dataset": dataset} for dataset in get_dataset_names()]}
