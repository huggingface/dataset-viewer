from typing import List, TypedDict

from datasets import list_datasets

from datasets_preview_backend.cache import cache, memoize  # type: ignore


class DatasetItem(TypedDict):
    dataset: str


class DatasetsContent(TypedDict):
    datasets: List[DatasetItem]


def get_dataset_items() -> List[DatasetItem]:
    # If an exception is raised, we let it propagate
    datasets: List[str] = list_datasets(with_community_datasets=True, with_details=False)  # type: ignore
    return [{"dataset": d} for d in datasets]


@memoize(cache)  # type:ignore
def get_datasets() -> DatasetsContent:
    return {"datasets": get_dataset_items()}
