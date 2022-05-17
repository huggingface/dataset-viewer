from admin.scripts.warm_cache import get_hf_dataset_names


# get_dataset_names
def test_get_hf_dataset_names() -> None:
    dataset_names = get_hf_dataset_names()
    assert len(dataset_names) > 1000
    assert "glue" in dataset_names
