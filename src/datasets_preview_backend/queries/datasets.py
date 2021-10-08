from typing import List, TypedDict

from datasets import list_datasets

from datasets_preview_backend.cache import cache, memoize  # type: ignore


class DatasetItem(TypedDict):
    dataset: str


class DatasetsContent(TypedDict):
    datasets: List[DatasetItem]


@memoize(cache=cache)  # type:ignore
def get_datasets() -> DatasetsContent:
    # If an exception is raised, we let starlette generate a 500 error
    datasets: List[str] = list_datasets(with_community_datasets=True, with_details=False)  # type: ignore
    return {"datasets": [{"dataset": d} for d in datasets]}
