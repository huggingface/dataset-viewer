from typing import List, TypedDict, Union

from datasets import list_datasets


class HFDataset(TypedDict):
    id: str
    tags: List[str]
    citation: Union[str, None]
    description: Union[str, None]
    paperswithcode_id: Union[str, None]
    downloads: Union[int, None]


def get_hf_datasets() -> List[HFDataset]:
    # If an exception is raised, we let it propagate
    datasets = list_datasets(with_community_datasets=True, with_details=True)  # type: ignore
    return [
        {
            "id": str(dataset.id),
            "tags": [str(tag) for tag in getattr(dataset, "tags", [])],
            "citation": getattr(dataset, "citation", None),
            "description": getattr(dataset, "description", None),
            "paperswithcode_id": getattr(dataset, "paperswithcode_id", None),
            "downloads": getattr(dataset, "downloads", None),
        }
        for dataset in datasets
    ]


def get_hf_dataset_names() -> List[str]:
    return [d["id"] for d in get_hf_datasets()]
