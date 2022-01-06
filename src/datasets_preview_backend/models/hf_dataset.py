import logging
from typing import List, TypedDict, Union

import requests
from datasets import list_datasets

logger = logging.getLogger(__name__)


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


def ask_access(dataset_name: str, hf_token: str) -> None:
    url = f"https://huggingface.co/datasets/{dataset_name}/ask-access"
    headers = {"Authorization": f"Bearer {hf_token}"}
    try:
        requests.get(url, headers=headers)
    except Exception as err:
        logger.warning(f"error while asking access to dataset {dataset_name}: {err}")
    # TODO: check if the access was granted: check if we were redirected to the dataset page, or to the login page


def get_hf_dataset_names() -> List[str]:
    return [d["id"] for d in get_hf_datasets()]
