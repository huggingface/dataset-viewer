from typing import List, TypedDict

from datasets_preview_backend.models.hf_dataset import get_hf_dataset_names


class DatasetItem(TypedDict):
    dataset: str


class DatasetsContent(TypedDict):
    datasets: List[DatasetItem]


def get_datasets() -> DatasetsContent:
    return {"datasets": [{"dataset": dataset_name} for dataset_name in get_hf_dataset_names()]}
