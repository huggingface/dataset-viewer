from datasets_preview_backend.queries.datasets import (
    get_datasets,
    get_refreshed_datasets,
)


def test_get_datasets() -> None:
    response = get_datasets()
    assert "datasets" in response
    datasets = response["datasets"]
    assert len(datasets) > 1000
    assert {"dataset": "glue"} in datasets


def test_get_refreshed_datasets() -> None:
    response = get_refreshed_datasets()
    assert "datasets" in response
    datasets = response["datasets"]
    assert len(datasets) > 1000
    assert {"dataset": "glue"} in datasets
