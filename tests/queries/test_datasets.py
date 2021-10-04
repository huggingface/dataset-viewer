from datasets_preview_backend.queries.datasets import get_datasets


def test_get_datasets() -> None:
    response = get_datasets()
    assert "datasets" in response
    datasets = response["datasets"]
    assert len(datasets) > 1000
    assert {"dataset": "glue"} in datasets
